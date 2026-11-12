import sys
import os
import types
import importlib.util
import shutil 
import json
import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog
from PIL import Image, ImageTk, ImageOps # NOVO: ImageOps para corrigir a rotação EXIF
import threading
import time

# =====================================================================
# 0. MONKEY PATCH (Correção obrigatória para Python 3.14+)
# =====================================================================
mock_pkg = types.ModuleType("pkg_resources")
def mock_resource_filename(pkg_name, resource_name):
    spec = importlib.util.find_spec(pkg_name)
    return os.path.join(os.path.dirname(spec.origin), resource_name)
mock_pkg.resource_filename = mock_resource_filename
sys.modules['pkg_resources'] = mock_pkg 

# =====================================================================
# 1. IMPORTAÇÕES DA VISÃO COMPUTACIONAL
# =====================================================================
try:
    import face_recognition
    import cv2
    import numpy as np
except Exception as e:
    import traceback
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Erro Fatal", f"Falha ao carregar motor principal:\n{e}")
    sys.exit(1)

# =====================================================================
# 2. ARQUITETURA DO SOFTWARE E DESIGN (UI/UX)
# =====================================================================
class BiometriaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema Neural de Biometria")
        
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
        self.cap = None
        self.loop_id = None
        self.is_camera_running = False
        self.frame_atual_camera = None
        
        self.vetores, self.nomes = [], []
        self.nome_cadastro = ""
        self.fotos_tiradas = 0
        self.diretorio_cadastro = ""
        
        self.is_verifying = False
        self.cadastro_iniciado = False
        self.rosto_alinhado = False
        self.cache_rostos_cadastro = []
        self.cache_resultados_ia = [] 
        
        # Variáveis de Simetria
        self.modo_simetria = "usuario" 
        self.caminho_upload_atual = ""
        
        self.arquivo_config = "config_biometria.json"
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
        self.is_camera_running = False 
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
                    return json.load(f).get("pasta_armazenamento", "dataset")
            except: return "dataset"
        return "dataset"

    def salvar_configuracao(self, novo_caminho):
        with open(self.arquivo_config, "w", encoding="utf-8") as f:
            json.dump({"pasta_armazenamento": novo_caminho}, f)

    def carregar_dados_biometricos(self):
        known_encodings, known_names = [], []
        if not os.path.exists(self.caminho_dataset): return known_encodings, known_names
        for nome in os.listdir(self.caminho_dataset):
            caminho_pessoa = os.path.join(self.caminho_dataset, nome)
            if not os.path.isdir(caminho_pessoa): continue
            encodings_pessoa = []
            for arquivo in os.listdir(caminho_pessoa):
                if arquivo.endswith(('.jpg', '.png')):
                    img = face_recognition.load_image_file(os.path.join(caminho_pessoa, arquivo))
                    encodes = face_recognition.face_encodings(img)
                    if len(encodes) > 0: encodings_pessoa.append(encodes[0])
            if len(encodings_pessoa) > 0:
                known_encodings.append(np.mean(encodings_pessoa, axis=0))
                known_names.append(nome)
        return known_encodings, known_names

    # ---------------- TELA 1: MENU PRINCIPAL ----------------
    def construir_menu(self):
        self.limpar_janela()
        card = tk.Frame(self.root, bg=self.bg_card, bd=0, highlightthickness=1, highlightbackground="#333333")
        card.place(relx=0.5, rely=0.5, anchor=tk.CENTER, width=500, height=720)

        tk.Label(card, text="⚙️", font=("Segoe UI", 36), bg=self.bg_card, fg=self.c_primary).pack(pady=(20, 0))
        tk.Label(card, text="Sistema de Biometria", font=self.f_title, bg=self.bg_card, fg=self.fg_text).pack()
        tk.Label(card, text="Autenticação Neural - Álgebra 4", font=self.f_subtitle, bg=self.bg_card, fg=self.fg_muted).pack(pady=(0, 20))

        self.criar_botao(card, "Inscrever Novo Usuário", self.c_success, self.c_success_h, self.iniciar_cadastro, icone="📸").pack(pady=10)
        self.criar_botao(card, "Verificar Acesso", self.c_primary, self.c_primary_h, self.iniciar_verificacao, icone="🔍").pack(pady=10)
        self.criar_botao(card, "Análise de Simetria Facial", self.c_cyan, self.c_cyan_h, self.iniciar_simetria_lista, icone="📐").pack(pady=10)
        self.criar_botao(card, "Gerenciar Usuários", self.c_purple, self.c_purple_h, self.iniciar_lista_usuarios, icone="👥").pack(pady=10)
        self.criar_botao(card, "Armazenamento", self.c_warning, self.c_warning_h, self.iniciar_configuracoes, icone="📁").pack(pady=10)
        
        btn_sair = tk.Button(card, text="Sair do Sistema", font=("Segoe UI", 10, "underline"), bg=self.bg_card, fg=self.c_danger, 
                             bd=0, activebackground=self.bg_card, activeforeground=self.c_danger_h, cursor="hand2", command=self.root.quit)
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
            fotos_disp = [f for f in os.listdir(caminho_pasta) if f.endswith(('.jpg', '.png'))]
            img_path = os.path.join(caminho_pasta, sorted(fotos_disp)[0])
        
        self.executar_preparativos_animacao(img_path)

    def executar_preparativos_animacao(self, img_path):
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
            txt_status = "📡 EXTRAINDO FOTO FRONTAL DO SERVIDOR..." if self.modo_simetria == "usuario" else "📂 CARREGANDO FOTO LOCAL DA MEMÓRIA..."
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
            image = face_recognition.load_image_file(img_path)
            landmarks_list = face_recognition.face_landmarks(image)
            
            if landmarks_list:
                marcas = landmarks_list[0]
                
                nariz_ponto = marcas['nose_bridge'][0] 
                olho_esq = np.mean(marcas['left_eye'], axis=0)
                olho_dir = np.mean(marcas['right_eye'], axis=0)
                mandibula_esq = marcas['chin'][2]
                mandibula_dir = marcas['chin'][14]
                
                dist_olho_esq = abs(nariz_ponto[0] - olho_esq[0])
                dist_olho_dir = abs(olho_dir[0] - nariz_ponto[0])
                diff_olhos = abs(dist_olho_esq - dist_olho_dir)
                
                dist_man_esq = abs(nariz_ponto[0] - mandibula_esq[0])
                dist_man_dir = abs(mandibula_dir[0] - nariz_ponto[0])
                diff_man = abs(dist_man_esq - dist_man_dir)
                
                largura_total = abs(marcas['chin'][16][0] - marcas['chin'][0][0])
                
                if largura_total > 0:
                    erro = diff_olhos + diff_man
                    pontuacao = max(0, 100 - ((erro / largura_total) * 100 * 2.5)) 
                    
                    self.resultado_matematico = {
                        "erro": False, 
                        "final": pontuacao, 
                        "dados": [{
                            "olhos": f"Esq: {dist_olho_esq:.1f}px | Dir: {dist_olho_dir:.1f}px",
                            "mandib": f"Esq: {dist_man_esq:.1f}px | Dir: {dist_man_dir:.1f}px",
                            "score": pontuacao
                        }]
                    }
                    return
        except: pass
            
        self.resultado_matematico = {"erro": True}

    # ---------------- TELA DE SIMETRIA 3: RESULTADO ÚNICO FRONTAL ----------------
    def mostrar_resultados_simetria(self):
        self.canvas.destroy() 
        
        tk.Label(self.root, text="✅ Análise Concluída", font=self.f_title, bg=self.bg_main, fg=self.c_success).pack(pady=(30, 10))
        
        if self.resultado_matematico["erro"]:
            tk.Label(self.root, text="Erro: Não foi possível detectar os vetores faciais (Landmarks) nesta foto.", bg=self.bg_main, fg=self.c_danger).pack()
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
        
        tk.Label(self.root, text="Grau de Simetria Bilateral Encontrado na Foto", font=self.f_normal, bg=self.bg_main, fg=self.fg_muted).pack()

        frame_fotos = tk.Frame(self.root, bg=self.bg_main)
        frame_fotos.pack(pady=30)
        
        if len(self.resultado_matematico["dados"]) > 0:
            dados = self.resultado_matematico["dados"][0]
            card = tk.Frame(frame_fotos, bg=self.bg_card, bd=2, relief=tk.RAISED)
            card.pack(padx=15)
            
            if self.fotos_animacao:
                tk.Label(card, image=self.fotos_animacao[0], bg=self.bg_card).pack(padx=20, pady=20)
            
            tk.Label(card, text="Distância Euclidiana Central", font=("Segoe UI", 11, "bold"), bg=self.bg_card, fg=self.c_primary).pack()
            tk.Label(card, text=f"Olhos:\n{dados['olhos']}", font=("Consolas", 10), bg=self.bg_card, fg="#FFFFFF").pack(pady=5)
            tk.Label(card, text=f"Maxilar:\n{dados['mandib']}", font=("Consolas", 10), bg=self.bg_card, fg="#FFFFFF").pack(pady=5, padx=20)

        self.criar_botao(self.root, "Concluir e Voltar", self.c_gray, self.c_gray_h, self.iniciar_simetria_lista, largura=20).pack(pady=20)

    def mostrar_teoria_matematica(self):
        teoria = (
            "CÁLCULO ALGÉBRICO DE SIMETRIA FACIAL\n\n"
            "O motor utiliza Análise de Landmarks para criar um vetor de 68 coordenadas (X, Y) do rosto humano.\n\n"
            "1. Eixo de Referência Central: \n"
            "Traçamos um eixo Y vertical imaginário a partir da ponte do nariz do usuário (ponto de origem central).\n\n"
            "2. Distância Euclidiana Bilateral: \n"
            "O algoritmo calcula o Módulo da Distância |x1 - x2| entre as extremidades esquerdas (olho e maxilar) "
            "e as extremidades direitas em relação ao eixo central.\n\n"
            "3. Relação de Erro: \n"
            "A diferença absoluta entre os lados esquerdo e direito é dividida pela largura total do rosto. "
            "Isso gera a taxa de assimetria. O resultado final que você vê na tela é: (100% - Margem de Assimetria)."
        )
        messagebox.showinfo("Teoria da Simetria - Álgebra 4", teoria)

    # ---------------- DEMAIS CONFIGURAÇÕES E CÂMERA MANTIDAS ----------------
    def iniciar_camera_assincrona(self):
        self.is_camera_running = True
        self.frame_atual_camera = None
        def leitor_camera():
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            while self.is_camera_running:
                ret, frame = self.cap.read()
                if ret: self.frame_atual_camera = cv2.flip(frame, 1)
                else: time.sleep(0.01)
            if self.cap is not None:
                self.cap.release()
                self.cap = None
        threading.Thread(target=leitor_camera, daemon=True).start()

    def exibir_tela_carregamento(self):
        img_loading = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img_loading, "Ativando sensor optico...", (180, 240), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 193, 7), 1)
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
        if not nome: return 
        self.nome_cadastro = nome.strip().title()
        self.diretorio_cadastro = os.path.join(self.caminho_dataset, self.nome_cadastro)
        self.fotos_tiradas, self.cadastro_iniciado, self.rosto_alinhado, self.cache_rostos_cadastro = 0, False, False, []
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
            self.criar_botao(self.frame_botoes_camera, f"Extrair Biometria ({self.fotos_tiradas}/3)", self.c_success, self.c_success_h, self.capturar_foto_cadastro, largura=25, icone="📸").pack(side=tk.LEFT, padx=15)
        self.criar_botao(self.frame_botoes_camera, "Cancelar", self.c_danger, self.c_danger_h, self.construir_menu, largura=20, icone="⏹️").pack(side=tk.LEFT, padx=15)

    def ativar_estado_captura(self):
        self.cadastro_iniciado = True
        self.atualizar_botoes_cadastro()
        threading.Thread(target=self.thread_ia_cadastro, daemon=True).start()

    def thread_ia_cadastro(self):
        while self.is_camera_running and self.cadastro_iniciado:
            if self.frame_atual_camera is not None:
                rgb_small = cv2.cvtColor(cv2.resize(self.frame_atual_camera, (0, 0), fx=0.25, fy=0.25), cv2.COLOR_BGR2RGB)
                self.cache_rostos_cadastro = face_recognition.face_locations(rgb_small, model="hog")
                time.sleep(0.05)
            else: time.sleep(0.02)

    def loop_cadastro(self):
        if not self.is_camera_running: return
        if self.frame_atual_camera is not None:
            frame = self.frame_atual_camera.copy()
            self.frame_atual_limpo = self.frame_atual_camera.copy() 
            h, w = frame.shape[:2]
            ox, oy, rx, ry = w // 2, h // 2, 120, 160 
            if not self.cadastro_iniciado:
                cv2.ellipse(frame, (ox, oy), (rx, ry), 0, 0, 360, (0, 255, 255), 2, cv2.LINE_AA)
                self.desenhar_texto_fundo_escuro(frame, "Clique no botao COMECAR abaixo", ox, h - 35, (0, 255, 255))
            else:
                self.rosto_alinhado = False
                if len(self.cache_rostos_cadastro) > 0:
                    t, r, b, l = [v*4 for v in self.cache_rostos_cadastro[0]]
                    fw, fh = r - l, b - t  
                    cx, cy = l + fw // 2, t + fh // 2 
                    if 120 < fw < 280 and abs(cx - ox) < 50 and abs(cy - oy) < 60: self.rosto_alinhado = True
                cor_molde = (0, 255, 0) if self.rosto_alinhado else (0, 0, 255)
                cv2.ellipse(frame, (ox, oy), (rx, ry), 0, 0, 360, cor_molde, 2, cv2.LINE_AA)
                instrucao = ""
                if self.rosto_alinhado:
                    if self.fotos_tiradas == 0: instrucao = "Olhe diretamente para a FRENTE"
                    elif self.fotos_tiradas == 1:
                        instrucao = "Vire o rosto para a DIREITA"
                        cv2.arrowedLine(frame, (ox + rx + 30, oy), (ox + rx + 100, oy), cor_molde, 5, cv2.LINE_AA, tipLength=0.3)
                    elif self.fotos_tiradas == 2:
                        instrucao = "Vire o rosto para a ESQUERDA"
                        cv2.arrowedLine(frame, (ox - rx - 30, oy), (ox - rx - 100, oy), cor_molde, 5, cv2.LINE_AA, tipLength=0.3)
                else: instrucao = "Encaixe o rosto no molde!"
                self.desenhar_texto_fundo_escuro(frame, instrucao, ox, h - 35, cor_molde)
            self.exibir_frame_tkinter(frame)
        else: self.exibir_tela_carregamento()
        self.loop_id = self.root.after(15, self.loop_cadastro)

    def desenhar_texto_fundo_escuro(self, frame, texto, x_centro, y, cor):
        texto_size = cv2.getTextSize(texto, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)[0]
        x_inicio = x_centro - texto_size[0] // 2
        cv2.rectangle(frame, (x_inicio - 10, y - 20), (x_inicio + texto_size[0] + 10, y + 10), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, texto, (x_inicio, y), cv2.FONT_HERSHEY_DUPLEX, 0.7, cor, 1, cv2.LINE_AA)

    def capturar_foto_cadastro(self):
        if not self.rosto_alinhado:
            messagebox.showwarning("Atenção", "O rosto está mal posicionado!\n\nPor favor, alinhe-se até que o molde fique VERDE.")
            return
        cv2.imwrite(os.path.join(self.diretorio_cadastro, f"foto_{self.fotos_tiradas+1}.jpg"), self.frame_atual_limpo)
        self.fotos_tiradas += 1
        if self.fotos_tiradas == 3:
            messagebox.showinfo("Operação Concluída", f"Matriz facial de {self.nome_cadastro} gerada e salva com sucesso.")
            self.construir_menu()
        else: self.atualizar_botoes_cadastro() 

    def iniciar_verificacao(self):
        self.vetores, self.nomes = self.carregar_dados_biometricos()
        if len(self.vetores) == 0:
            messagebox.showwarning("Aviso", "O banco de dados está vazio. Cadastre alguém primeiro.")
            return
        self.limpar_janela()
        frame_botoes = self.construir_tela_camera("🔍 Escaneamento em Tempo Real")
        self.criar_botao(frame_botoes, "Encerrar Câmera", self.c_danger, self.c_danger_h, self.construir_menu, largura=20, icone="⏹️").pack(side=tk.LEFT, padx=15)
        self.is_verifying, self.cache_resultados_ia = True, []
        self.iniciar_camera_assincrona()
        threading.Thread(target=self.thread_ia_reconhecimento, daemon=True).start()
        self.loop_verificacao()

    def thread_ia_reconhecimento(self):
        while self.is_verifying:
            if self.frame_atual_camera is not None:
                frame_atual = self.frame_atual_camera.copy()
                rgb_small = cv2.cvtColor(cv2.resize(frame_atual, (0, 0), fx=0.25, fy=0.25), cv2.COLOR_BGR2RGB)
                locs = face_recognition.face_locations(rgb_small, model="hog")
                if len(locs) > 0:
                    encodings = face_recognition.face_encodings(rgb_small, locs)
                    resultados = []
                    for i, encoding in enumerate(encodings):
                        dist = face_recognition.face_distance(self.vetores, encoding)
                        nome = "ACESSO NEGADO"
                        if len(dist) > 0 and dist[np.argmin(dist)] <= 0.5: nome = self.nomes[np.argmin(dist)]
                        resultados.append((locs[i], nome))
                    self.cache_resultados_ia = resultados
                else: self.cache_resultados_ia = [] 
                time.sleep(0.05) 
            else: time.sleep(0.02)

    def loop_verificacao(self):
        if not self.is_verifying: return
        if self.frame_atual_camera is not None:
            frame = self.frame_atual_camera.copy()
            for (top, right, bottom, left), name in self.cache_resultados_ia:
                t, r, b, l = top * 4, right * 4, bottom * 4, left * 4
                cor = (0, 255, 0) if name != "ACESSO NEGADO" else (0, 0, 255)
                L, thickness = 20, 3
                for (x1,y1, x2,y2) in [(l,t, l+L,t), (l,t, l,t+L), (r,t, r-L,t), (r,t, r,t+L), (l,b, l+L,b), (l,b, l,b-L), (r,b, r-L,b), (r,b, r,b-L)]:
                    cv2.line(frame, (x1, y1), (x2, y2), cor, thickness)
                overlay = frame.copy()
                cv2.rectangle(overlay, (l, b), (r, b + 35), cor, cv2.FILLED)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                cv2.putText(frame, name.upper(), (l + 8, b + 24), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            self.exibir_frame_tkinter(frame)
        else: self.exibir_tela_carregamento()
        self.loop_id = self.root.after(15, self.loop_verificacao)

    def exibir_frame_tkinter(self, frame_cv2):
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB)))
        self.lbl_video.imgtk = imgtk 
        self.lbl_video.configure(image=imgtk)

if __name__ == "__main__":
    app_root = tk.Tk()
    app = BiometriaApp(app_root)
    app_root.mainloop()