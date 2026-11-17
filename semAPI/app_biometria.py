import sys
import os
import shutil 
import json
import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog, ttk
from PIL import Image, ImageTk, ImageOps # NOVO: ImageOps para corrigir a rotação EXIF
import threading
import time
import queue
from concurrent.futures import CancelledError

# =====================================================================
# 0. CAMINHOS DO APLICATIVO
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(
    sys.executable if getattr(sys, "frozen", False) else __file__
))

# =====================================================================
# 1. IMPORTAÇÕES DA VISÃO COMPUTACIONAL
# =====================================================================
try:
    import cv2
    import numpy as np
    from reconhecimento_classico import (
        EXTENSOES, carregar_referencias, classificar, extrair_descritor, ler_imagem,
        normalizar_rosto, salvar_olhos,
        distancia_m2hd, salvar_painel_contornos,
    )
except Exception as e:
    import traceback
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Erro Fatal", f"Falha ao carregar motor principal:\n{e}")
    sys.exit(1)

# =====================================================================
# 2. ARQUITETURA DO SOFTWARE E DESIGN (UI/UX)
# =====================================================================
def calcular_simetria_manual(pontos):
    """Medida 2D didática por seis marcações, sem detector ou modelo treinado.

    Ordem: olhos à esquerda/direita da foto, ponte do nariz, ponta do queixo,
    extremidades à esquerda/direita. Os olhos definem o referencial; entram
    na nota os erros de reflexão do contorno, do nariz e do queixo.
    """
    pontos = np.asarray(pontos, dtype=np.float64)
    if pontos.shape != (6, 2) or not np.isfinite(pontos).all():
        raise ValueError("Marque os seis pontos pedidos na fotografia.")
    horizontal = pontos[1] - pontos[0]
    distancia_olhos = float(np.linalg.norm(horizontal))
    if not np.isfinite(distancia_olhos) or distancia_olhos < 1e-6 or horizontal[0] <= 0:
        raise ValueError("Marque os dois olhos separados, da esquerda para a direita da foto.")
    horizontal /= distancia_olhos
    vertical = np.array([-horizontal[1], horizontal[0]])
    origem = (pontos[0] + pontos[1]) / 2
    alinhados = (pontos - origem) @ np.column_stack((horizontal, vertical))
    olho_esq, olho_dir, nariz, queixo, extremo_esq, extremo_dir = alinhados
    largura = float(extremo_dir[0] - extremo_esq[0])
    if not np.isfinite(alinhados).all() or not np.isfinite(largura):
        raise ValueError("As coordenadas das marcações são inválidas.")
    if largura < 1e-6 or not extremo_esq[0] < 0 < extremo_dir[0]:
        raise ValueError("Marque uma extremidade em cada lado do rosto, seguindo os lados da foto.")
    if queixo[1] <= max(0., nariz[1]):
        raise ValueError("A ponta do queixo deve ficar abaixo dos olhos e do nariz no rosto alinhado.")
    if not extremo_esq[0] <= queixo[0] <= extremo_dir[0]:
        raise ValueError("A ponta do queixo deve ficar entre as duas extremidades do rosto.")
    refletir = np.array([-1., 1.])
    erros = {
        "contorno": float(np.linalg.norm(extremo_esq * refletir - extremo_dir) / largura),
        "nariz": float(np.linalg.norm(nariz * refletir - nariz) / largura),
        "queixo": float(np.linalg.norm(queixo * refletir - queixo) / largura),
    }
    return {
        "pontuacao": float(np.clip(100 * (1 - np.mean(list(erros.values()))), 0, 100)),
        "largura": largura,
        "distancias_olhos": (float(-olho_esq[0]), float(olho_dir[0])),
        "distancias_extremidades": (float(-extremo_esq[0]), float(extremo_dir[0])),
        "desnivel_extremidades": float(abs(extremo_esq[1] - extremo_dir[1])),
        "desvio_nariz": float(abs(nariz[0])),
        "queixo_alinhado": (float(queixo[0]), float(queixo[1])),
        "erros_normalizados": erros,
    }


class BiometriaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Biometria Clássica — Contornos e Hausdorff")
        self.root.protocol("WM_DELETE_WINDOW", self.fechar)
        
        # 🎨 PALETA DE CORES (Dark Mode Profissional)
        self.bg_main = "#121212"      
        self.bg_card = "#1E1E1E"      
        self.fg_text = "#FFFFFF"      
        self.fg_muted = "#A0A0A0"     
        
        self.c_primary = "#2196F3"    
        self.c_primary_h = "#1976D2"  
        self.c_success = "#4CAF50"    
        self.c_success_h = "#388E3C"  
        self.c_danger = "#F44336"     
        self.c_danger_h = "#D32F2F"   
        self.c_purple = "#9C27B0"     
        self.c_purple_h = "#7B1FA2"   
        self.c_warning = "#FF9800"    
        self.c_warning_h = "#F57C00"  
        self.c_gray = "#424242"       
        self.c_gray_h = "#616161"
        self.c_cyan = "#00BCD4"       
        self.c_cyan_h = "#0097A7"
        
        # 🖋️ FONTES
        self.f_title = ("Segoe UI", 22, "bold")
        self.f_subtitle = ("Segoe UI", 12)
        self.f_btn = ("Segoe UI", 12, "bold")
        self.f_normal = ("Segoe UI", 11)

        self.root.configure(bg=self.bg_main)
        
        largura_janela, altura_janela = 900, 750 
        largura_tela = self.root.winfo_screenwidth()
        altura_tela = self.root.winfo_screenheight()
        pos_x = int((largura_tela / 2) - (largura_janela / 2))
        pos_y = int((altura_tela / 2) - (altura_janela / 2))
        self.root.geometry(f"{largura_janela}x{altura_janela}+{pos_x}+{pos_y}")
        
        # Variáveis Globais
        self.loop_id = None
        self.is_camera_running = False
        self.frame_atual_camera = None
        self.erro_camera = None
        self.parar_camera = threading.Event()
        self.thread_camera = None
        self.sessao_camera = 0
        self.parar_processamento = threading.Event()
        self.loop_processamento = None
        self.thread_processamento = None
        
        self.referencias, self.nomes = [], []
        self.nome_cadastro = ""
        self.diretorio_cadastro = ""
        
        self.is_verifying = False
        self.cadastro_iniciado = False
        self.resultado_comparacao = None
        self.captura_verificacao = None
        self.comparando = False
        self.detalhes_comparacao = None
        self.origens_referencias = []
        self.imagem_resultado = None
        self.mapas_comparacao = None
        
        # Variáveis de Simetria
        self.modo_simetria = "usuario" 
        self.caminho_upload_atual = ""
        
        self.arquivo_config = os.path.join(BASE_DIR, "config_biometria.json")
        self.caminho_dataset = self.carregar_configuracao()
        if not os.path.exists(self.caminho_dataset):
            os.makedirs(self.caminho_dataset)
        
        self.construir_menu()

    def criar_botao(self, parent, texto, cor, cor_hover, comando, largura=25, icone=""):
        texto_final = f"{icone}  {texto}" if icone else texto
        btn = tk.Button(parent, text=texto_final, font=self.f_btn, bg=cor, fg="#FFFFFF", 
                        activebackground=cor_hover, activeforeground="#FFFFFF",
                        bd=0, cursor="hand2", width=largura, pady=12, command=comando)
        btn.bind("<Enter>", lambda e: btn.config(bg=cor_hover))
        btn.bind("<Leave>", lambda e: btn.config(bg=cor))
        return btn

    def limpar_janela(self):
        self.is_verifying = False 
        self.cadastro_iniciado = False
        self.encerrar_camera()
        self.sessao_camera += 1
        self.comparando = False
        self.parar_processamento.set()
        self.video_verificacao = False
        if self.loop_processamento is not None:
            self.root.after_cancel(self.loop_processamento)
            self.loop_processamento = None
        if self.loop_id is not None:
            self.root.after_cancel(self.loop_id)
            self.loop_id = None
        for widget in self.root.winfo_children():
            widget.destroy()

    # ---------------- LÓGICA DE CONFIGURAÇÃO E DADOS ----------------
    def carregar_configuracao(self):
        if os.path.exists(self.arquivo_config):
            try:
                with open(self.arquivo_config, "r", encoding="utf-8") as f:
                    caminho = json.load(f).get("pasta_armazenamento", "dataset")
                    return os.path.abspath(os.path.join(BASE_DIR, caminho))
            except (OSError, ValueError, TypeError): pass
        return os.path.join(BASE_DIR, "dataset")

    def salvar_configuracao(self, novo_caminho):
        with open(self.arquivo_config, "w", encoding="utf-8") as f:
            json.dump({"pasta_armazenamento": novo_caminho}, f)

    def carregar_dados_biometricos(self):
        referencias, nomes, avisos = carregar_referencias(
            self.caminho_dataset, self.marcar_olhos)
        if avisos:
            messagebox.showwarning("Referências não carregadas", "\n".join(avisos[:12]))
        return referencias, nomes

    def marcar_olhos(self, imagem, titulo="Marcar olhos"):
        return self.selecionar_pontos(imagem, ("olho à esquerda na foto", "olho à direita na foto"), titulo)

    def executar_processamento(self, tarefa, concluir):
        """Cálculos fora do Tkinter; progresso, diálogos e resultado na thread da UI."""
        self.parar_processamento.set()
        parar = self.parar_processamento = threading.Event()
        sessao = self.sessao_camera
        mensagens = queue.Queue()
        self.comparando = True
        self.inicio_processamento = time.monotonic()

        def progresso(fracao, texto):
            if not parar.is_set():
                mensagens.put(("progresso", (fracao, texto)))

        def marcar(imagem, titulo):
            pronto, resposta = threading.Event(), []
            mensagens.put(("marcar", (imagem, titulo, pronto, resposta)))
            while not pronto.wait(.05):
                if parar.is_set():
                    raise CancelledError()
            return resposta[0]

        def calcular():
            try:
                resultado = tarefa(progresso, parar, marcar)
                if not parar.is_set():
                    mensagens.put(("concluido", resultado))
            except CancelledError:
                pass
            except Exception as erro:
                import traceback
                traceback.print_exc()
                mensagens.put(("erro", str(erro) or type(erro).__name__))

        def acompanhar():
            self.loop_processamento = None
            if parar.is_set() or sessao != self.sessao_camera:
                return
            acabou = False
            while not mensagens.empty():
                tipo, dados = mensagens.get_nowait()
                if tipo == "progresso":
                    fracao, texto = dados
                    self.barra_progresso.config(value=100 * fracao)
                    self.lbl_andamento.config(text=texto)
                elif tipo == "marcar":
                    imagem, titulo, pronto, resposta = dados
                    try:
                        resposta.append(self.marcar_olhos(imagem, titulo))
                    finally:
                        pronto.set()
                    if parar.is_set() or sessao != self.sessao_camera:
                        return
                elif tipo in ("concluido", "erro"):
                    acabou = True
                    self.comparando = False
                    self.barra_progresso.config(value=100 if tipo == "concluido" else 0)
                    if tipo == "erro":
                        self.lbl_comparacao.config(text="Não foi possível concluir a comparação", fg=self.c_warning)
                        self.lbl_andamento.config(text=f"{dados}\nAjuste a captura e tente novamente.")
                        self.btn_comparar.config(state=tk.NORMAL if self.referencias else tk.DISABLED)
                        self.btn_nova_captura.config(state=tk.NORMAL)
                    else:
                        concluir(dados)
                    break
            if not acabou:
                decorrido = time.monotonic() - self.inicio_processamento
                self.lbl_tempo.config(text=f"Processando há {decorrido:.0f} s")
                self.loop_processamento = self.root.after(40, acompanhar)
            else:
                self.lbl_tempo.config(text="")

        self.thread_processamento = threading.Thread(target=calcular, daemon=True)
        self.thread_processamento.start()
        acompanhar()

    def selecionar_pontos(self, imagem, rotulos, titulo, instrucoes=None):
        """Marcações humanas; não há detector facial ou de landmarks oculto."""
        janela = tk.Toplevel(self.root)
        janela.title(titulo)
        janela.configure(bg=self.bg_main)
        janela.transient(self.root)
        janela.grab_set()
        if instrucoes:
            tk.Label(janela, text=instrucoes, wraplength=680, justify=tk.LEFT,
                     font=("Segoe UI", 10), bg=self.bg_main, fg=self.fg_muted).pack(padx=15, pady=(10, 0))
        altura, largura = imagem.shape[:2]
        escala = min(720 / largura, 450 / altura, 1.)
        w, h = round(largura * escala), round(altura * escala)
        aviso = tk.Label(janela, font=self.f_normal, bg=self.bg_main, fg=self.fg_text)
        aviso.pack(padx=15, pady=10)
        canvas = tk.Canvas(janela, width=w, height=h, highlightthickness=0, cursor="crosshair")
        canvas.pack(padx=15)
        foto = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)).resize((w, h)))
        canvas.create_image(0, 0, anchor=tk.NW, image=foto)
        pontos, resultado = [], []

        def atualizar():
            canvas.delete("marcas")
            for i, (x, y) in enumerate(pontos):
                px, py = x * w / largura, y * h / altura
                canvas.create_oval(px-4, py-4, px+4, py+4, outline="#00FFFF", width=2, tags="marcas")
                canvas.create_text(px+12, py-10, text=str(i+1), fill="#00FFFF", tags="marcas")
            aviso.config(text=f"Clique no centro: {rotulos[len(pontos)]}" if len(pontos) < len(rotulos)
                         else "Confira os pontos e confirme. As marcações não serão gravadas na foto.")
            confirmar.config(state=tk.NORMAL if len(pontos) == len(rotulos) else tk.DISABLED)

        def clicar(evento):
            if len(pontos) < len(rotulos) and 0 <= evento.x < w and 0 <= evento.y < h:
                pontos.append((evento.x * largura / w, evento.y * altura / h))
                atualizar()

        def desfazer():
            if pontos:
                pontos.pop()
                atualizar()

        def concluir():
            if len(pontos) == len(rotulos):
                resultado.extend(pontos)
                janela.destroy()

        botoes = tk.Frame(janela, bg=self.bg_main)
        botoes.pack(pady=15)
        confirmar = tk.Button(botoes, text="Confirmar", command=concluir)
        confirmar.pack(side=tk.LEFT, padx=10)
        tk.Button(botoes, text="Desfazer ponto", command=desfazer).pack(side=tk.LEFT, padx=10)
        tk.Button(botoes, text="Cancelar", command=janela.destroy).pack(side=tk.LEFT, padx=10)
        canvas.bind("<Button-1>", clicar)
        janela.bind("<Escape>", lambda e: janela.destroy())
        atualizar()
        self.root.wait_window(janela)
        return resultado or None

    # ---------------- TELA 1: MENU PRINCIPAL ----------------
    def construir_menu(self):
        self.limpar_janela()
        card = tk.Frame(self.root, bg=self.bg_card, bd=0, highlightthickness=1, highlightbackground="#333333")
        card.place(relx=0.5, rely=0.5, anchor=tk.CENTER, width=500, height=720)

        tk.Label(card, text="⚙️", font=("Segoe UI", 36), bg=self.bg_card, fg=self.c_primary).pack(pady=(20, 0))
        tk.Label(card, text="Sistema de Biometria", font=self.f_title, bg=self.bg_card, fg=self.fg_text).pack()
        tk.Label(card, text="Método clássico — pontos dominantes e M²HD", font=self.f_subtitle, bg=self.bg_card, fg=self.fg_muted).pack(pady=(0, 20))

        self.criar_botao(card, "Inscrever Novo Usuário", self.c_success, self.c_success_h, self.iniciar_cadastro, icone="📸").pack(pady=10)
        self.criar_botao(card, "Reconhecer por Contornos", self.c_primary, self.c_primary_h, self.iniciar_verificacao, icone="🔍").pack(pady=10)
        self.criar_botao(card, "Análise de Simetria Facial", self.c_cyan, self.c_cyan_h, self.iniciar_simetria_lista, icone="📐").pack(pady=10)
        self.criar_botao(card, "Gerenciar Usuários", self.c_purple, self.c_purple_h, self.iniciar_lista_usuarios, icone="👥").pack(pady=10)
        self.criar_botao(card, "Armazenamento", self.c_warning, self.c_warning_h, self.iniciar_configuracoes, icone="📁").pack(pady=10)
        
        btn_sair = tk.Button(card, text="Sair do Sistema", font=("Segoe UI", 10, "underline"), bg=self.bg_card, fg=self.c_danger, 
                             bd=0, activebackground=self.bg_card, activeforeground=self.c_danger_h, cursor="hand2", command=self.fechar)
        btn_sair.pack(pady=20)

    # ---------------- TELA DE SIMETRIA 1: LISTA E UPLOAD ----------------
    def iniciar_simetria_lista(self):
        self.limpar_janela()
        tk.Label(self.root, text="📐 Análise de Simetria Facial", font=self.f_title, bg=self.bg_main, fg=self.fg_text).pack(pady=(40, 5))
        tk.Label(self.root, text="Selecione um usuário ou faça upload de uma foto avulsa.", font=self.f_normal, bg=self.bg_main, fg=self.fg_muted).pack(pady=(0, 15))

        frame_upload = tk.Frame(self.root, bg=self.bg_main)
        frame_upload.pack(pady=5)
        self.criar_botao(frame_upload, "Fazer Upload de Foto Avulsa", self.c_warning, self.c_warning_h, self.iniciar_tela_upload_simetria, largura=35, icone="📤").pack()

        card = tk.Frame(self.root, bg=self.bg_card)
        card.pack(fill=tk.BOTH, expand=True, padx=80, pady=15)

        if not os.path.exists(self.caminho_dataset): os.makedirs(self.caminho_dataset)
        usuarios = [d for d in os.listdir(self.caminho_dataset) if os.path.isdir(os.path.join(self.caminho_dataset, d))]

        if not usuarios:
            tk.Label(card, text="Nenhum rosto cadastrado no banco de dados.", font=self.f_normal, bg=self.bg_card, fg=self.fg_muted).pack(pady=40)
            self.criar_botao(card, "Cadastrar um Usuário", self.c_success, self.c_success_h, self.iniciar_cadastro, largura=25, icone="➕").pack(pady=10)
        else:
            for nome in sorted(usuarios):
                row = tk.Frame(card, bg="#2A2A2A", bd=0)
                row.pack(fill=tk.X, padx=20, pady=5)
                tk.Label(row, text=f"👤  {nome}", font=self.f_btn, bg="#2A2A2A", fg=self.fg_text, padx=15, pady=12).pack(side=tk.LEFT)
                btn_calc = tk.Button(row, text="🧠 Calcular Simetria", font=("Segoe UI", 10, "bold"), bg=self.c_cyan, fg="#FFFFFF", bd=0,
                                    activebackground=self.c_cyan_h, activeforeground="#FFFFFF", cursor="hand2", padx=10,
                                    command=lambda n=nome: self.preparar_animacao_simetria(n))
                btn_calc.pack(side=tk.RIGHT, padx=10, pady=8)
        
        self.criar_botao(self.root, "Voltar ao Menu", self.c_gray, self.c_gray_h, self.construir_menu, largura=20, icone="🔙").pack(pady=30)

    # ---------------- TELA DE SIMETRIA: DROPZONE DE UPLOAD AVULSO ----------------
    def iniciar_tela_upload_simetria(self):
        self.limpar_janela()
        tk.Label(self.root, text="📤 Upload de Foto Avulsa", font=self.f_title, bg=self.bg_main, fg=self.fg_text).pack(pady=(40, 10))
        tk.Label(self.root, text="Selecione uma imagem do seu computador para análise isolada.\n(A imagem não será salva no banco de dados)", font=self.f_normal, bg=self.bg_main, fg=self.fg_muted, justify=tk.CENTER).pack(pady=(0, 20))
        
        self.frame_dropzone = tk.Frame(self.root, bg=self.bg_card, highlightbackground=self.c_primary, highlightthickness=2, bd=0)
        self.frame_dropzone.pack(pady=20, padx=80, fill=tk.BOTH, expand=True)
        
        lbl_drop_icon = tk.Label(self.frame_dropzone, text="📁", font=("Segoe UI", 48), bg=self.bg_card, fg=self.c_primary, cursor="hand2")
        lbl_drop_icon.pack(pady=(80, 10))
        lbl_drop_text = tk.Label(self.frame_dropzone, text="Clique aqui para procurar a imagem\n(Formatos aceitos: JPG, PNG, WEBP, JPEG, BMP)", font=self.f_normal, bg=self.bg_card, fg=self.fg_muted, cursor="hand2")
        lbl_drop_text.pack(pady=(0, 80))
        
        lbl_drop_icon.bind("<Button-1>", lambda e: self.selecionar_foto_upload())
        lbl_drop_text.bind("<Button-1>", lambda e: self.selecionar_foto_upload())
        self.frame_dropzone.bind("<Button-1>", lambda e: self.selecionar_foto_upload())
        
        self.frame_botoes_upload = tk.Frame(self.root, bg=self.bg_main)
        self.frame_botoes_upload.pack(pady=20)
        
        self.criar_botao(self.frame_botoes_upload, "Voltar", self.c_gray, self.c_gray_h, self.iniciar_simetria_lista, largura=20, icone="🔙").pack(side=tk.LEFT, padx=10)

    def selecionar_foto_upload(self):
        caminho = filedialog.askopenfilename(
            title="Selecione a Imagem",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.webp *.bmp")]
        )
        if caminho:
            self.caminho_upload_atual = caminho
            
            for widget in self.frame_dropzone.winfo_children():
                widget.destroy()
            
            try:
                img = Image.open(caminho)
                # OTIMIZAÇÃO: Corrige a orientação EXIF que o celular salva (horizontal vs vertical)
                img = ImageOps.exif_transpose(img) 
                
                img.thumbnail((300, 300))
                self.preview_upload_tk = ImageTk.PhotoImage(img)
                
                tk.Label(self.frame_dropzone, image=self.preview_upload_tk, bg=self.bg_card).pack(pady=20)
                tk.Label(self.frame_dropzone, text="Imagem carregada com sucesso na memória!", font=self.f_btn, bg=self.bg_card, fg=self.c_success).pack()
                
                for widget in self.frame_botoes_upload.winfo_children():
                    widget.destroy()
                
                self.criar_botao(self.frame_botoes_upload, "Continuar Análise", self.c_cyan, self.c_cyan_h, self.preparar_animacao_simetria_upload, largura=25, icone="▶️").pack(side=tk.LEFT, padx=10)
                self.criar_botao(self.frame_botoes_upload, "Cancelar", self.c_danger, self.c_danger_h, self.iniciar_simetria_lista, largura=20, icone="⏹️").pack(side=tk.LEFT, padx=10)
                
            except Exception as e:
                messagebox.showerror("Erro", f"Não foi possível ler o formato desta imagem:\n{e}")
                self.iniciar_tela_upload_simetria()

    # ---------------- TELA DE SIMETRIA 2: ANIMAÇÃO DE IMAGEM ÚNICA ----------------
    def preparar_animacao_simetria_upload(self):
        self.modo_simetria = "upload"
        self.executar_preparativos_animacao(self.caminho_upload_atual)
        
    def preparar_animacao_simetria(self, nome_usuario):
        self.modo_simetria = "usuario"
        caminho_pasta = os.path.join(self.caminho_dataset, nome_usuario)
        img_path = os.path.join(caminho_pasta, "foto_1.jpg")
        if not os.path.exists(img_path): 
            fotos_disp = [f for f in os.listdir(caminho_pasta) if os.path.splitext(f)[1].lower() in EXTENSOES]
            if not fotos_disp:
                messagebox.showwarning("Sem fotos", "Conclua o cadastro de uma foto antes de analisar a simetria.")
                return
            img_path = os.path.join(caminho_pasta, sorted(fotos_disp)[0])
        
        self.executar_preparativos_animacao(img_path)

    def executar_preparativos_animacao(self, img_path):
        try:
            imagem = ler_imagem(img_path)
        except (OSError, ValueError) as erro:
            messagebox.showerror("Imagem inválida", str(erro))
            return
        self.pontos_simetria = self.selecionar_pontos(imagem,
            ("olho à esquerda na foto", "olho à direita na foto", "ponte do nariz",
             "ponta do queixo", "extremidade do rosto à esquerda na foto",
             "extremidade do rosto à direita na foto"),
            "Simetria — marcar seis pontos",
            instrucoes="Use uma foto frontal, com expressão neutra. Marque dois pontos equivalentes "
                       "nas laterais do contorno do rosto. Esquerda e direita são os lados da foto.")
        if self.pontos_simetria is None:
            return
        try:
            calcular_simetria_manual(self.pontos_simetria)
        except ValueError as erro:
            messagebox.showwarning("Confira as marcações", str(erro))
            return
        self.limpar_janela()
        self.canvas = tk.Canvas(self.root, bg=self.bg_main, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.fotos_animacao = []
        try:
            img_main = Image.open(img_path)
            # OTIMIZAÇÃO: Corrige a orientação EXIF que o celular salva (Garante o escaneamento certo)
            img_main = ImageOps.exif_transpose(img_main)
            
            img_thumb = img_main.copy()
            img_thumb.thumbnail((200, 200)) 
            self.fotos_animacao.append(ImageTk.PhotoImage(img_thumb))
            
            img_main.thumbnail((360, 480))
            self.foto_scan_tk = ImageTk.PhotoImage(img_main)
            self.foto_scan_w, self.foto_scan_h = img_main.size
        except:
            self.foto_scan_tk = None
            self.foto_scan_w, self.foto_scan_h = 300, 400
        
        self.anim_fase = 1
        self.anim_y_fotos = 600
        self.anim_y_laser = 0
        self.laser_dir = 1
        self.start_time_scan = None
        
        self.resultado_matematico = None
        threading.Thread(target=self.thread_calcular_simetria_matematica, args=(img_path,), daemon=True).start()
        
        self.loop_animacao_simetria()

    def loop_animacao_simetria(self):
        self.canvas.delete("all")
        w, h = self.root.winfo_width(), self.root.winfo_height()
        rx, ry = w//2, h//2
        
        if self.anim_fase == 1:
            txt_status = "📂 CARREGANDO FOTO FRONTAL LOCAL..." if self.modo_simetria == "usuario" else "📂 CARREGANDO FOTO LOCAL DA MEMÓRIA..."
            self.canvas.create_text(rx, 100, text=txt_status, fill=self.c_cyan, font=("Consolas", 16, "bold"))
            
            self.canvas.create_rectangle(rx - 60, 150, rx + 60, 230, fill="#2A2A2A", outline=self.c_cyan, width=3)
            self.canvas.create_line(rx - 40, 170, rx + 40, 170, fill=self.c_primary, width=4)
            self.canvas.create_line(rx - 40, 190, rx + 40, 190, fill=self.c_primary, width=4)
            
            self.anim_y_fotos -= 20
            
            if self.fotos_animacao:
                self.canvas.create_image(rx, self.anim_y_fotos, image=self.fotos_animacao[0])
                self.canvas.create_line(rx, self.anim_y_fotos - 100, rx, 230, fill="#4CAF50", dash=(4,4))
                
            if self.anim_y_fotos < 330:
                self.anim_fase = 2
                
        elif self.anim_fase == 2:
            self.canvas.create_text(rx, 80, text="⚙️ CALCULANDO DISTÂNCIAS EUCLIDIANAS...", fill=self.c_warning, font=("Consolas", 16, "bold"))
            
            if self.start_time_scan is None:
                self.start_time_scan = time.time()
                
            elapsed = time.time() - self.start_time_scan
            percentual = min(100, int((elapsed / 10.0) * 100))
            
            topo_img = ry - self.foto_scan_h//2
            esq_img = rx - self.foto_scan_w//2
            dir_img = rx + self.foto_scan_w//2
            
            if self.foto_scan_tk:
                self.canvas.create_image(rx, ry, image=self.foto_scan_tk)
                
                velocidade_laser = 15
                self.anim_y_laser += velocidade_laser * self.laser_dir
                if self.anim_y_laser > self.foto_scan_h: self.laser_dir = -1
                if self.anim_y_laser < 0: self.laser_dir = 1
                
                y_atual_laser = topo_img + self.anim_y_laser
                
                self.canvas.create_line(esq_img - 15, y_atual_laser, dir_img + 15, y_atual_laser, fill="#00FF00", width=4)
                self.canvas.create_line(esq_img - 5, y_atual_laser-6, dir_img + 5, y_atual_laser-6, fill="#00FF00", width=1)
                
                hud_x = esq_img - 130
                angulo_giro = (elapsed * 120) % 360
                self.canvas.create_arc(hud_x-60, ry-60, hud_x+60, ry+60, start=angulo_giro, extent=270, outline=self.c_success, width=3, style=tk.ARC)
                self.canvas.create_oval(hud_x-50, ry-50, hud_x+50, ry+50, outline=self.c_success, width=1, dash=(2,4))
                self.canvas.create_text(hud_x, ry, text=f"{percentual}%", fill=self.c_success, font=("Consolas", 24, "bold"))
                self.canvas.create_line(hud_x+60, ry, esq_img, y_atual_laser, fill=self.c_success, dash=(2,2))
                
                rad_x = dir_img + 130
                self.canvas.create_oval(rad_x-50, ry-50, rad_x+50, ry+50, outline=self.c_cyan, width=2)
                self.canvas.create_oval(rad_x-15, ry-15, rad_x+15, ry+15, fill=self.c_cyan)
                linha_radar = (angulo_giro * -1.5) % 360
                self.canvas.create_arc(rad_x-50, ry-50, rad_x+50, ry+50, start=linha_radar, extent=45, fill=self.c_cyan, outline="")
                self.canvas.create_text(rad_x, ry-75, text="ANÁLISE\nBIOMÉTRICA", fill=self.c_cyan, font=("Consolas", 12, "bold"), justify=tk.CENTER)
                self.canvas.create_line(rad_x-50, ry, dir_img, y_atual_laser, fill=self.c_cyan, dash=(2,2))

            if elapsed >= 10.0 and self.resultado_matematico is not None:
                self.anim_fase = 3

        if self.anim_fase < 3:
            self.loop_id = self.root.after(40, self.loop_animacao_simetria)
        else:
            self.mostrar_resultados_simetria()

    # ---------------- LÓGICA MATEMÁTICA CENTRAL (REUTILIZÁVEL) ----------------
    def thread_calcular_simetria_matematica(self, img_path):
        try:
            medidas = calcular_simetria_manual(self.pontos_simetria)
            olho_esq, olho_dir = medidas["distancias_olhos"]
            extremo_esq, extremo_dir = medidas["distancias_extremidades"]
            queixo_x, queixo_y = medidas["queixo_alinhado"]
            self.resultado_matematico = {
                "erro": False, "final": medidas["pontuacao"], "medidas": medidas,
                "dados": [{
                    "olhos": f"Esq: {olho_esq:.1f}px | Dir: {olho_dir:.1f}px (referência)",
                    "extremidades": f"Esq: {extremo_esq:.1f}px | Dir: {extremo_dir:.1f}px\n"
                                    f"Desnível: {medidas['desnivel_extremidades']:.1f}px",
                    "nariz": f"Desvio do eixo: {medidas['desvio_nariz']:.1f}px",
                    "queixo": f"Desvio do eixo: {abs(queixo_x):.1f}px\n"
                              f"Altura abaixo dos olhos: {queixo_y:.1f}px",
                    "largura": f"{medidas['largura']:.1f}px",
                }],
            }
        except ValueError as erro:
            self.resultado_matematico = {"erro": True, "mensagem": str(erro)}

    # ---------------- TELA DE SIMETRIA 3: RESULTADO ÚNICO FRONTAL ----------------
    def mostrar_resultados_simetria(self):
        self.canvas.destroy() 
        
        tk.Label(self.root, text="✅ Análise Concluída", font=self.f_title, bg=self.bg_main, fg=self.c_success).pack(pady=(30, 10))
        
        if self.resultado_matematico["erro"]:
            tk.Label(self.root, text=self.resultado_matematico.get("mensagem", "Confira as seis marcações."),
                     wraplength=760, bg=self.bg_main, fg=self.c_danger).pack()
            self.criar_botao(self.root, "Voltar", self.c_gray, self.c_gray_h, self.iniciar_simetria_lista).pack(pady=20)
            return

        nota_final = self.resultado_matematico["final"]
        cor_nota = self.c_success if nota_final > 90 else self.c_warning if nota_final > 80 else self.c_danger
        
        frame_placar = tk.Frame(self.root, bg=self.bg_main)
        frame_placar.pack(pady=10)
        
        tk.Label(frame_placar, text=f"{nota_final:.1f}%", font=("Segoe UI", 48, "bold"), bg=self.bg_main, fg=cor_nota).pack(side=tk.LEFT, padx=10)
        
        btn_teoria = tk.Button(frame_placar, text="?", font=("Arial", 16, "bold"), bg="#333333", fg="white", bd=0, 
                               cursor="hand2", width=3, command=self.mostrar_teoria_matematica)
        btn_teoria.pack(side=tk.LEFT, padx=10)
        
        tk.Label(self.root, text="Estimativa geométrica de simetria com seis pontos", font=self.f_normal,
                 bg=self.bg_main, fg=self.fg_muted).pack()

        frame_fotos = tk.Frame(self.root, bg=self.bg_main)
        frame_fotos.pack(pady=15)
        
        if len(self.resultado_matematico["dados"]) > 0:
            dados = self.resultado_matematico["dados"][0]
            card = tk.Frame(frame_fotos, bg=self.bg_card, bd=2, relief=tk.RAISED)
            card.pack(padx=15)
            
            if self.fotos_animacao:
                tk.Label(card, image=self.fotos_animacao[0], bg=self.bg_card).pack(side=tk.LEFT, padx=15, pady=15)
            detalhes = tk.Frame(card, bg=self.bg_card)
            detalhes.pack(side=tk.LEFT, padx=15, pady=12)
            
            tk.Label(detalhes, text="Medidas no rosto alinhado", font=("Segoe UI", 11, "bold"),
                     bg=self.bg_card, fg=self.c_primary).pack(anchor="w", pady=(0, 5))
            for rotulo, chave in (("Olhos", "olhos"), ("Extremidades", "extremidades"),
                                  ("Nariz", "nariz"), ("Queixo", "queixo"), ("Largura facial", "largura")):
                tk.Label(detalhes, text=f"{rotulo}: {dados[chave]}", justify=tk.LEFT, wraplength=490,
                         font=("Consolas", 10), bg=self.bg_card, fg="#FFFFFF").pack(anchor="w", pady=3)

        tk.Label(self.root, text="Use foto frontal e pontos equivalentes nas laterais. "
                 "A pontuação é didática, não uma avaliação clínica.", wraplength=780,
                 font=("Segoe UI", 10), bg=self.bg_main, fg=self.fg_muted).pack(pady=5)

        self.criar_botao(self.root, "Concluir e Voltar", self.c_gray, self.c_gray_h, self.iniciar_simetria_lista, largura=20).pack(pady=20)

    def mostrar_teoria_matematica(self):
        teoria = (
            "CÁLCULO ALGÉBRICO DE SIMETRIA FACIAL\n\n"
            "O usuário marca seis pontos: dois olhos, ponte do nariz, ponta do queixo e duas laterais do rosto.\n"
            "Esta ferramenta de simetria é separada do reconhecimento por M²HD.\n\n"
            "1. Eixo de Referência Central: \n"
            "A linha dos olhos corrige a inclinação. Sua metade define a origem, e o eixo perpendicular "
            "é a referência central. Os olhos servem ao alinhamento, não recebem uma nota própria.\n\n"
            "2. Reflexão e distâncias em duas dimensões:\n"
            "Refletimos uma extremidade (x, y) em (-x, y) e medimos a distância até a outra. "
            "Também comparamos nariz e queixo com suas próprias posições refletidas. "
            "A altura do queixo é mostrada; mudar somente sua altura não indica assimetria bilateral.\n\n"
            "3. Normalização e pontuação:\n"
            "Dividimos os três erros pela largura facial e calculamos sua média. "
            "Pontuação = máximo(0, 100 × (1 - erro médio)). "
            "É uma medida geométrica didática, sensível às marcações, pose e expressão."
        )
        messagebox.showinfo("Teoria da Simetria - Álgebra 4", teoria)

    # ---------------- DEMAIS CONFIGURAÇÕES E CÂMERA MANTIDAS ----------------
    def iniciar_camera_assincrona(self):
        self.encerrar_camera()
        if self.thread_camera is not None:
            self.thread_camera.join(timeout=1)
        self.is_camera_running = True
        self.frame_atual_camera = None
        self.erro_camera = None
        self.parar_camera = threading.Event()
        parar = self.parar_camera
        def leitor_camera():
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            try:
                if not cap.isOpened():
                    if not parar.is_set():
                        self.erro_camera = "Camera indisponivel. Verifique a conexao."
                    return
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                while not parar.is_set():
                    ret, frame = cap.read()
                    if ret and not parar.is_set():
                        self.frame_atual_camera = cv2.flip(frame, 1)
                    elif not ret:
                        if not parar.is_set():
                            self.erro_camera = "Falha na leitura da camera. Reabra a captura."
                            self.frame_atual_camera = None
                        return
            finally:
                cap.release()
        self.thread_camera = threading.Thread(target=leitor_camera, daemon=True)
        self.thread_camera.start()

    def encerrar_camera(self):
        self.is_camera_running = False
        self.parar_camera.set()

    def fechar(self):
        self.limpar_janela()
        self.root.destroy()

    def exibir_tela_carregamento(self):
        img_loading = np.zeros((480, 640, 3), dtype=np.uint8)
        texto = self.erro_camera or "Ativando sensor optico..."
        cv2.putText(img_loading, texto, (20, 240), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 193, 7), 1)
        self.exibir_frame_tkinter(img_loading)

    def iniciar_configuracoes(self):
        self.limpar_janela()
        tk.Label(self.root, text="📁 Configuração de Diretório", font=self.f_title, bg=self.bg_main, fg=self.fg_text).pack(pady=(40, 10))
        card = tk.Frame(self.root, bg=self.bg_card, bd=0)
        card.pack(fill=tk.BOTH, expand=False, padx=80, pady=20)
        tk.Label(card, text="Selecione o local no seu computador onde as matrizes faciais serão salvas.", font=self.f_normal, bg=self.bg_card, fg=self.fg_muted).pack(pady=20)
        frame_path = tk.Frame(card, bg="#000000", bd=0)
        frame_path.pack(fill=tk.X, padx=40, pady=10)
        self.lbl_caminho = tk.Label(frame_path, text=os.path.abspath(self.caminho_dataset), font=("Consolas", 10), bg="#000000", fg="#4CAF50", padx=15, pady=15, wraplength=600)
        self.lbl_caminho.pack()
        self.criar_botao(card, "Alterar Diretório", self.c_primary, self.c_primary_h, self.escolher_nova_pasta, largura=20).pack(pady=30)
        self.criar_botao(self.root, "Voltar ao Menu", self.c_gray, self.c_gray_h, self.construir_menu, largura=20, icone="🔙").pack(pady=20)

    def escolher_nova_pasta(self):
        novo_caminho = filedialog.askdirectory(title="Selecione a pasta do Banco de Dados")
        if novo_caminho: 
            self.caminho_dataset = novo_caminho
            self.lbl_caminho.config(text=os.path.abspath(self.caminho_dataset))
            self.salvar_configuracao(self.caminho_dataset)

    def iniciar_lista_usuarios(self):
        self.limpar_janela()
        tk.Label(self.root, text="👥 Gerenciamento de Identidades", font=self.f_title, bg=self.bg_main, fg=self.fg_text).pack(pady=(40, 20))
        card = tk.Frame(self.root, bg=self.bg_card)
        card.pack(fill=tk.BOTH, expand=True, padx=80, pady=10)
        if not os.path.exists(self.caminho_dataset): os.makedirs(self.caminho_dataset)
        usuarios = [d for d in os.listdir(self.caminho_dataset) if os.path.isdir(os.path.join(self.caminho_dataset, d))]
        if not usuarios:
            tk.Label(card, text="Nenhum usuário consta no banco de dados.", font=self.f_normal, bg=self.bg_card, fg=self.fg_muted).pack(pady=40)
        else:
            for nome in sorted(usuarios):
                row = tk.Frame(card, bg="#2A2A2A", bd=0)
                row.pack(fill=tk.X, padx=20, pady=5)
                tk.Label(row, text=f"👤  {nome}", font=self.f_btn, bg="#2A2A2A", fg=self.fg_text, padx=15, pady=12).pack(side=tk.LEFT)
                btn_del = tk.Button(row, text="🗑️ Remover", font=("Segoe UI", 10, "bold"), bg=self.c_danger, fg="#FFFFFF", bd=0,
                                    activebackground=self.c_danger_h, activeforeground="#FFFFFF", cursor="hand2", padx=10, command=lambda n=nome: self.confirmar_exclusao(n))
                btn_del.pack(side=tk.RIGHT, padx=10, pady=8)
                btn_del.bind("<Enter>", lambda e, b=btn_del: b.config(bg=self.c_danger_h))
                btn_del.bind("<Leave>", lambda e, b=btn_del: b.config(bg=self.c_danger))
        self.criar_botao(self.root, "Voltar ao Menu", self.c_gray, self.c_gray_h, self.construir_menu, largura=20, icone="🔙").pack(pady=30)

    def confirmar_exclusao(self, nome_usuario):
        if messagebox.askokcancel("Confirmar Deleção", f"Isso revogará permanentemente o acesso de '{nome_usuario}'.\n\nDeseja prosseguir?"):
            try:
                shutil.rmtree(os.path.join(self.caminho_dataset, nome_usuario))
                self.iniciar_lista_usuarios()
            except Exception as e: messagebox.showerror("Erro", f"Falha no sistema de arquivos: {e}")

    def construir_tela_camera(self, titulo_texto):
        tk.Label(self.root, text=titulo_texto, font=self.f_title, bg=self.bg_main, fg=self.fg_text).pack(pady=(20, 10))
        moldura_video = tk.Frame(self.root, bg=self.c_primary, bd=2)
        moldura_video.pack(pady=10)
        self.lbl_video = tk.Label(moldura_video, bg="black")
        self.lbl_video.pack(padx=2, pady=2)
        frame_botoes = tk.Frame(self.root, bg=self.bg_main)
        frame_botoes.pack(pady=20)
        return frame_botoes

    def iniciar_cadastro(self):
        nome = simpledialog.askstring("Identificação", "Insira o nome completo do novo usuário:")
        if not nome or not nome.strip(): return
        self.nome_cadastro = nome.strip().title()
        if self.nome_cadastro in (".", "..") or any(c in self.nome_cadastro for c in '<>:"/\\|?*'):
            messagebox.showwarning("Nome inválido", "Use um nome sem caracteres de caminho ou de arquivo.")
            return
        self.diretorio_cadastro = os.path.join(self.caminho_dataset, self.nome_cadastro)
        self.cadastro_iniciado = False
        if not os.path.exists(self.diretorio_cadastro): os.makedirs(self.diretorio_cadastro)
        self.limpar_janela()
        self.frame_botoes_camera = self.construir_tela_camera(f"Cadastrando: {self.nome_cadastro}")
        self.atualizar_botoes_cadastro()
        self.iniciar_camera_assincrona()
        self.loop_cadastro()

    def atualizar_botoes_cadastro(self):
        for widget in self.frame_botoes_camera.winfo_children(): widget.destroy()
        if not self.cadastro_iniciado:
            self.criar_botao(self.frame_botoes_camera, "Começar", self.c_primary, self.c_primary_h, self.ativar_estado_captura, largura=25, icone="▶️").pack(side=tk.LEFT, padx=15)
        else:
            self.criar_botao(self.frame_botoes_camera, "Capturar Foto Frontal", self.c_success, self.c_success_h, self.capturar_foto_cadastro, largura=25, icone="📸").pack(side=tk.LEFT, padx=15)
        self.criar_botao(self.frame_botoes_camera, "Cancelar", self.c_danger, self.c_danger_h, self.construir_menu, largura=20, icone="⏹️").pack(side=tk.LEFT, padx=15)

    def ativar_estado_captura(self):
        self.cadastro_iniciado = True
        self.atualizar_botoes_cadastro()

    def loop_cadastro(self):
        if not self.is_camera_running: return
        if self.frame_atual_camera is not None:
            frame = self.frame_atual_camera.copy()
            h, w = frame.shape[:2]
            ox = w // 2
            if not self.cadastro_iniciado:
                self.desenhar_molde_rosto(frame, (0, 255, 255))
                self.desenhar_texto_fundo_escuro(frame, "Clique no botao COMECAR abaixo", ox, h - 35, (0, 255, 255))
            else:
                cor_molde = (255, 255, 0)
                self.desenhar_molde_rosto(frame, cor_molde)
                instrucao = "Capture uma foto frontal e marque os olhos"
                self.desenhar_texto_fundo_escuro(frame, instrucao, ox, h - 35, cor_molde)
            self.exibir_frame_tkinter(frame)
        else: self.exibir_tela_carregamento()
        self.loop_id = self.root.after(15, self.loop_cadastro)

    def desenhar_molde_rosto(self, frame, cor=(255, 255, 0)):
        """Guia visual comum ao cadastro e reconhecimento, desenhada só na prévia."""
        h, w = frame.shape[:2]
        escala = min(w / 640, h / 480)
        raios = (max(1, round(120 * escala)), max(1, round(160 * escala)))
        cv2.ellipse(frame, (w // 2, h // 2), raios, 0, 0, 360, cor, 2, cv2.LINE_AA)

    def desenhar_texto_fundo_escuro(self, frame, texto, x_centro, y, cor):
        texto_size = cv2.getTextSize(texto, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)[0]
        x_inicio = x_centro - texto_size[0] // 2
        cv2.rectangle(frame, (x_inicio - 10, y - 20), (x_inicio + texto_size[0] + 10, y + 10), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, texto, (x_inicio, y), cv2.FONT_HERSHEY_DUPLEX, 0.7, cor, 1, cv2.LINE_AA)

    def capturar_foto_cadastro(self):
        if self.frame_atual_camera is None:
            messagebox.showwarning("Câmera", "Aguarde uma imagem válida da câmera.")
            return
        imagem = self.frame_atual_camera.copy()
        olhos = self.marcar_olhos(imagem, "Cadastro — alinhar foto frontal")
        if olhos is None:
            return
        try:
            extrair_descritor(normalizar_rosto(imagem, olhos))
            indice = 1
            while os.path.exists(os.path.join(self.diretorio_cadastro, f"foto_{indice}.jpg")):
                indice += 1
            caminho = os.path.join(self.diretorio_cadastro, f"foto_{indice}.jpg")
            # Salva a captura limpa, nunca as guias, pontos ou texto da tela.
            Image.fromarray(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)).save(caminho, quality=95)
            salvar_olhos(caminho, olhos)
        except (OSError, ValueError, cv2.error) as erro:
            messagebox.showerror("Captura não salva", str(erro))
            return
        messagebox.showinfo("Cadastro concluído", f"A foto frontal de {self.nome_cadastro} foi salva com sucesso.")
        self.construir_menu()

    def iniciar_verificacao(self):
        self.limpar_janela()
        self.referencias, self.nomes, self.origens_referencias = [], [], []
        self.detalhes_comparacao = None
        self.mapas_comparacao = None
        self.captura_verificacao = self.imagem_resultado = None
        self.resultado_comparacao = None
        self.video_verificacao = True
        tk.Label(self.root, text="Reconhecimento por contornos", font=self.f_title,
                 bg=self.bg_main, fg=self.fg_text).pack(pady=(15, 5))
        # O resultado vem antes do vídeo, sem depender do espaço que sobrar na janela.
        self.lbl_comparacao = tk.Label(self.root, text="Preparando as fotos cadastradas...",
            bg=self.bg_main, fg=self.fg_text, font=("Segoe UI", 16, "bold"), wraplength=850)
        self.lbl_comparacao.pack(pady=5)
        self.lbl_andamento = tk.Label(self.root, text="A câmera pode ser posicionada enquanto a base carrega.",
            bg=self.bg_main, fg=self.fg_muted, font=self.f_normal, wraplength=850)
        self.lbl_andamento.pack()
        self.barra_progresso = ttk.Progressbar(self.root, maximum=100, length=550)
        self.barra_progresso.pack(pady=5)
        self.lbl_tempo = tk.Label(self.root, text="", bg=self.bg_main, fg=self.fg_muted)
        self.lbl_tempo.pack()
        self.lbl_video = tk.Label(self.root, bg="black")
        self.lbl_video.pack(pady=5)
        frame_botoes = tk.Frame(self.root, bg=self.bg_main)
        frame_botoes.pack(pady=10)
        self.btn_comparar = self.criar_botao(frame_botoes, "Comparar captura", self.c_primary, self.c_primary_h,
                         self.comparar_captura, largura=20, icone="📸")
        self.btn_comparar.pack(side=tk.LEFT, padx=10)
        self.btn_nova_captura = self.criar_botao(frame_botoes, "Nova captura", self.c_gray, self.c_gray_h,
                         self.nova_captura, largura=15)
        self.btn_nova_captura.pack(side=tk.LEFT, padx=10)
        self.criar_botao(frame_botoes, "Voltar", self.c_danger, self.c_danger_h,
                         self.construir_menu, largura=12).pack(side=tk.LEFT, padx=10)
        frame_exportacao = tk.Frame(self.root, bg=self.bg_main)
        frame_exportacao.pack()
        self.btn_detalhes = tk.Button(frame_exportacao, text="Ver detalhes da comparação", command=self.mostrar_detalhes_comparacao,
            bg=self.bg_card, fg=self.fg_text, bd=0, padx=15, pady=5, state=tk.DISABLED)
        self.btn_detalhes.pack(side=tk.LEFT, padx=5)
        self.btn_salvar_contornos = tk.Button(frame_exportacao, text="Salvar contornos", command=self.salvar_contornos_comparacao,
            bg=self.bg_card, fg=self.fg_text, bd=0, padx=15, pady=5, state=tk.DISABLED)
        self.btn_salvar_contornos.pack(side=tk.LEFT, padx=5)
        self.btn_comparar.config(state=tk.DISABLED)
        self.btn_nova_captura.config(state=tk.DISABLED)
        self.is_verifying = True
        self.iniciar_camera_assincrona()
        self.loop_verificacao()

        def carregar(progresso, parar, marcar):
            origens = []
            referencias, nomes, avisos = carregar_referencias(self.caminho_dataset,
                marcar_olhos=marcar, progresso=progresso, cancelar=parar, origens=origens)
            return referencias, nomes, avisos, origens

        def base_pronta(dados):
            self.referencias, self.nomes, avisos, self.origens_referencias = dados
            self.btn_nova_captura.config(state=tk.NORMAL)
            if self.referencias:
                self.btn_comparar.config(state=tk.NORMAL)
                self.lbl_comparacao.config(text="Pronto para comparar", fg=self.fg_text)
                self.lbl_andamento.config(text=f"{len(set(self.nomes))} pessoa(s), {len(self.referencias)} foto(s). "
                    "Encaixe o rosto inteiro no molde antes de comparar."
                    + (f" {len(avisos)} foto(s) ignorada(s)." if avisos else ""))
            else:
                self.lbl_comparacao.config(text="Nenhuma foto válida disponível", fg=self.c_warning)
                self.lbl_andamento.config(text="Cadastre fotos frontais para começar. " + (avisos[0] if avisos else ""))
        self.executar_processamento(carregar, base_pronta)

    def nova_captura(self):
        if self.comparando:
            return
        self.captura_verificacao = None
        self.resultado_comparacao = None
        self.imagem_resultado = self.detalhes_comparacao = None
        self.mapas_comparacao = None
        self.btn_detalhes.config(state=tk.DISABLED)
        self.btn_salvar_contornos.config(state=tk.DISABLED)
        self.barra_progresso.config(value=0)
        if self.referencias:
            self.lbl_comparacao.config(text="Pronto para comparar", fg=self.fg_text)
            self.lbl_andamento.config(text="Encaixe o rosto inteiro no molde. Afaste-se se necessário e clique em Comparar captura.")
        else:
            self.lbl_comparacao.config(text="Nenhuma foto válida disponível", fg=self.c_warning)
            self.lbl_andamento.config(text="Volte ao menu e cadastre fotos frontais para começar.")

    def comparar_captura(self):
        if self.comparando or not self.referencias:
            return
        if self.frame_atual_camera is None:
            self.lbl_comparacao.config(text="Aguardando imagem da câmera", fg=self.c_warning)
            self.lbl_andamento.config(text=self.erro_camera or "Aguarde a câmera iniciar e tente novamente.")
            return
        imagem = self.frame_atual_camera.copy()
        olhos = self.marcar_olhos(imagem, "Reconhecimento — alinhar captura")
        if olhos is None:
            return
        self.captura_verificacao = imagem
        self.resultado_comparacao = None
        self.imagem_resultado = self.detalhes_comparacao = None
        self.mapas_comparacao = None
        referencias, nomes = self.referencias, self.nomes
        origens = self.origens_referencias
        self.btn_comparar.config(state=tk.DISABLED)
        self.btn_nova_captura.config(state=tk.DISABLED)
        self.btn_detalhes.config(state=tk.DISABLED)
        self.btn_salvar_contornos.config(state=tk.DISABLED)
        self.lbl_comparacao.config(text="Analisando a captura...", fg=self.fg_text)
        self.lbl_andamento.config(text="Alinhando o rosto e extraindo seus contornos.")
        self.barra_progresso.config(value=0)

        def calcular(progresso, parar, marcar):
            recorte = normalizar_rosto(imagem, olhos)
            descritor = extrair_descritor(recorte,
                progresso=lambda f, t: progresso(.8 * f, t), cancelar=parar)
            ranking = classificar(descritor, referencias, nomes,
                progresso=lambda f, t: progresso(.8 + .2 * f, t), cancelar=parar)
            if not ranking:
                raise ValueError("A base não contém fotos válidas para a comparação.")
            indice = min((i for i, nome in enumerate(nomes) if nome == ranking[0][0]),
                         key=lambda i: distancia_m2hd(referencias[i], descritor))
            referencia = None
            if indice < len(origens):
                origem = origens[indice]
                try:
                    referencia = normalizar_rosto(ler_imagem(origem["caminho"]), origem["olhos"])
                except (OSError, ValueError, cv2.error):
                    pass
            return ranking, len(descritor), recorte, referencia, descritor, referencias[indice]

        def mostrar(dados):
            ranking, quantidade, recorte, referencia, descritor, descritor_referencia = dados
            self.resultado_comparacao = (ranking, quantidade, None)
            self.detalhes_comparacao = dados[:2]
            self.lbl_comparacao.config(text=f"Melhor correspondência: {ranking[0][0]}", fg=self.c_cyan)
            self.lbl_andamento.config(text="Comparação concluída. Veja a captura e a foto cadastrada abaixo.")
            self.btn_comparar.config(state=tk.NORMAL)
            self.btn_nova_captura.config(state=tk.NORMAL)
            self.btn_detalhes.config(state=tk.NORMAL)
            self.mapas_comparacao = [("Captura analisada", recorte, descritor)]
            if referencia is not None:
                self.mapas_comparacao.append((f"Referência: {ranking[0][0]}", referencia, descritor_referencia))
            self.btn_salvar_contornos.config(state=tk.NORMAL)
            painel = np.full((240, 340, 3), 24, dtype=np.uint8)
            painel[35:235, :160] = recorte
            if referencia is not None:
                painel[35:235, 180:340] = referencia
            cv2.putText(painel, "Captura", (4, 23), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            cv2.putText(painel, "Foto cadastrada" if referencia is not None else "Sem previa",
                        (182, 23), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
            self.imagem_resultado = cv2.resize(painel, (476, 336))
            self.exibir_frame_tkinter(self.imagem_resultado)
        self.executar_processamento(calcular, mostrar)

    def salvar_contornos_comparacao(self):
        if not self.mapas_comparacao:
            return
        pasta = os.path.join(BASE_DIR, "exportacoes")
        try:
            os.makedirs(pasta, exist_ok=True)
            caminho = filedialog.asksaveasfilename(title="Salvar contornos e pontos dominantes",
                initialdir=pasta, initialfile="comparacao_contornos.png", defaultextension=".png",
                filetypes=[("Imagem PNG", "*.png")])
            if not caminho:
                return
            salvar_painel_contornos(caminho, self.mapas_comparacao)
            messagebox.showinfo("Contornos salvos", f"Mapa dos contornos e pontos dominantes salvo em:\n{caminho}")
        except (OSError, ValueError, cv2.error) as erro:
            messagebox.showerror("Falha na exportação", str(erro))

    def mostrar_detalhes_comparacao(self):
        if self.detalhes_comparacao is None:
            return
        ranking, quantidade = self.detalhes_comparacao
        texto = f"{quantidade} pontos dominantes na captura.\n\n"
        texto += "\n".join(f"{i}. {nome}: {distancia:.2f} pixels" for i, (nome, distancia) in enumerate(ranking, 1))
        texto += ("\n\nMenor distância M²HD indica contornos mais próximos. "
                  "A distância não é uma porcentagem de confiança. "
                  "O ranking compara os cadastros disponíveis; não confirma que uma pessoa desconhecida pertence à base.")
        messagebox.showinfo("Detalhes da comparação geométrica", texto)

    def loop_verificacao(self):
        if not self.is_verifying: return
        if self.imagem_resultado is not None:
            pass  # Resultado estático já renderizado; não converter a mesma imagem a cada quadro.
        elif self.captura_verificacao is not None or self.frame_atual_camera is not None:
            frame = (self.captura_verificacao if self.captura_verificacao is not None
                     else self.frame_atual_camera).copy()
            if self.captura_verificacao is None:
                self.desenhar_molde_rosto(frame)
                h, w = frame.shape[:2]
                self.desenhar_texto_fundo_escuro(frame, "Encaixe o rosto inteiro no molde", w // 2, h - 55, (255, 255, 0))
                self.desenhar_texto_fundo_escuro(frame, "Afaste-se se o rosto ultrapassar a guia", w // 2, h - 25, (255, 255, 0))
            self.exibir_frame_tkinter(frame)
        else: self.exibir_tela_carregamento()
        self.loop_id = self.root.after(15, self.loop_verificacao)

    def exibir_frame_tkinter(self, frame_cv2):
        imagem = Image.fromarray(cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB))
        imagem.thumbnail((640, 340) if getattr(self, "video_verificacao", False) else (640, 480))
        imgtk = ImageTk.PhotoImage(image=imagem)
        self.lbl_video.imgtk = imgtk 
        self.lbl_video.configure(image=imgtk)

if __name__ == "__main__":
    app_root = tk.Tk()
    app = BiometriaApp(app_root)
    app_root.mainloop()
    app.encerrar_camera()
