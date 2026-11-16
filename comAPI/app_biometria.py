"""Ponto de entrada: interface Tkinter e inicialização automática da API local."""

from io import BytesIO
import importlib.util
from pathlib import Path
from queue import Empty, Full, Queue
import secrets
import socket
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import traceback
import unicodedata
from urllib.parse import quote

BASE = Path(__file__).resolve().parent

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageOps, ImageTk
    import requests
except ImportError as erro:
    comando = f'& "{sys.executable}" -m pip install -r "{BASE / "requirements.txt"}"'
    print(f"Dependência ausente: {erro}\nInstale usando o mesmo Python:\n{comando}")
    janela = tk.Tk()
    janela.withdraw()
    messagebox.showerror("Dependências", f"{erro}\n\nSelecione o ambiente comAPI/.venv no VS Code.\nComando de instalação:\n{comando}")
    janela.destroy()
    raise SystemExit(1)


def jpeg(imagem):
    sucesso, dados = cv2.imencode(".jpg", imagem, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not sucesso:
        raise ValueError("Não foi possível preparar a foto.")
    return dados.tobytes()


class MalhaEmMovimento:
    """Fluxo óptico só para o desenho; nunca produz identidade ou biometria."""

    def __init__(self):
        self.cinza = self.pontos = None
        self.tamanhos = []
        self.instante = 0
        self.validade = 1.5
        self.ausencias = 0
        self.falha_desde = None

    @staticmethod
    def cinza_reduzido(frame):
        h, w = frame.shape[:2]
        pequeno = cv2.resize(frame, (320, max(1, round(h * 320 / w))), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(pequeno, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def rastrear(antes, atual, pontos, tamanhos):
        novos, status, erros = cv2.calcOpticalFlowPyrLK(
            antes, atual, pontos, None, winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        if novos is None or status is None or erros is None:
            return None
        validos = status.ravel().astype(bool) & np.isfinite(novos.reshape(-1, 2)).all(axis=1)
        validos &= erros.ravel() < 25
        deslocamento = 0
        for tamanho in tamanhos:
            trecho = slice(deslocamento, deslocamento + tamanho)
            bons = validos[trecho]
            # Pontos sobre pele lisa podem não ter textura para fluxo óptico.
            # Usa os pontos confiáveis para mover a malha inteira, sem apagá-la
            # porque uma pequena região não pôde ser acompanhada.
            if bons.sum() < max(12, tamanho * 0.2):
                return None
            origem = pontos[trecho, 0]
            destino = novos[trecho, 0]
            matriz, inliers = cv2.estimateAffinePartial2D(
                origem[bons], destino[bons], method=cv2.RANSAC,
                ransacReprojThreshold=2, maxIters=100)
            if matriz is None or inliers is None or inliers.mean() < 0.5:
                return None
            estimados = cv2.transform(origem.reshape(-1, 1, 2), matriz)[:, 0]
            confiaveis = bons & (np.linalg.norm(destino - estimados, axis=1) < 3)
            estimados[confiaveis] = destino[confiaveis]
            novos[trecho, 0] = estimados
            deslocamento += tamanho
        return novos

    def atualizar(self, frame, malhas, instante, recebido_em=None, validade=1.5):
        agora = instante if recebido_em is None else recebido_em
        self.validade = validade
        if not malhas:
            self.ausencias += 1
            if self.ausencias >= 2 or agora - self.instante > self.validade:
                self.pontos = self.cinza = None
            return
        self.ausencias = 0
        cinza = self.cinza_reduzido(frame)
        h, w = frame.shape[:2]
        tamanhos = [len(m) for m in malhas]
        escala = np.array([320 / w, cinza.shape[0] / h], dtype=np.float32)
        pontos = (np.concatenate(malhas).astype(np.float32) * escala).reshape(-1, 1, 2)
        if self.cinza is not None and self.cinza.shape == cinza.shape:
            # A resposta HTTP descreve um quadro antigo. Traz a medição para
            # o último quadro exibido antes de trocar o estado do rastreador.
            alinhados = self.rastrear(cinza, self.cinza, pontos, tamanhos)
            if alinhados is not None:
                pontos, cinza = alinhados, self.cinza
                if self.pontos is not None and tamanhos == self.tamanhos and len(tamanhos) == 1:
                    if np.median(np.linalg.norm(pontos - self.pontos, axis=2)) < 15:
                        pontos = 0.7 * pontos + 0.3 * self.pontos
        self.pontos, self.cinza, self.tamanhos = pontos, cinza, tamanhos
        self.instante, self.falha_desde = agora, None

    def acompanhar(self, frame, agora):
        if self.pontos is None or agora - self.instante > self.validade:
            return []
        atual = self.cinza_reduzido(frame)
        if atual.shape != self.cinza.shape:
            self.pontos = None
            return []
        novos = self.rastrear(self.cinza, atual, self.pontos, self.tamanhos)
        if novos is None:
            if self.falha_desde is None:
                self.falha_desde = agora
            if agora - self.falha_desde > 0.6:
                self.pontos = None
                return []
            # Aguarda brevemente a próxima medição real sem piscar o desenho.
            novos = self.pontos
        else:
            self.cinza, self.pontos = atual, novos
            self.falha_desde = None
        h, w = frame.shape[:2]
        escala = np.array([w / atual.shape[1], h / atual.shape[0]], dtype=np.float32)
        malhas, deslocamento = [], 0
        for tamanho in self.tamanhos:
            trecho = slice(deslocamento, deslocamento + tamanho)
            depois = novos[trecho, 0]
            malhas.append(depois * escala)
            deslocamento += tamanho
        return malhas


class LeituraVisual:
    """Retenção visual breve; a captura continua usando a análise bruta e recente."""

    def __init__(self):
        self.faces = []
        self.recebido_em = self.ultimo_resultado = None
        self.intervalo = 0.6
        self.validade = 1.5
        self.ausencias = 0

    def atualizar(self, faces, agora):
        if self.ultimo_resultado is not None:
            intervalo = agora - self.ultimo_resultado
            self.intervalo = 0.7 * self.intervalo + 0.3 * intervalo
        self.ultimo_resultado = agora
        self.validade = min(3.0, max(1.5, 2.5 * self.intervalo))
        if faces:
            self.faces, self.recebido_em, self.ausencias = faces, agora, 0
        else:
            self.ausencias += 1
            if self.ausencias >= 2:
                self.faces = []

    def visiveis(self, agora):
        if self.recebido_em is None or agora - self.recebido_em > self.validade:
            return []
        return self.faces


class Camera:
    """Leitura contínua; a análise sempre pega a captura mais recente."""

    def __init__(self):
        self.parar = threading.Event()
        self.lock = threading.Lock()
        self.frame = None
        self.sequencia = 0
        self.erro = None
        threading.Thread(target=self.ler, daemon=True).start()

    def ler(self):
        cap = None
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
            if not cap.isOpened():
                raise RuntimeError("Não foi possível abrir a webcam. Confira as permissões e outros aplicativos.")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            while not self.parar.is_set():
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("A webcam deixou de entregar imagens.")
                # Sem espelhar: preserva os lados anatômicos e as fotos originais.
                with self.lock:
                    self.frame = frame
                    self.sequencia += 1
        except Exception as erro:
            self.erro = str(erro)
        finally:
            if cap is not None:
                cap.release()

    def foto(self):
        with self.lock:
            return (self.frame.copy() if self.frame is not None else None), self.sequencia


class BiometriaApp:
    BG = "#121212"
    CARD = "#1E1E1E"
    MUTED = "#A0A0A0"

    def __init__(self, root):
        self.root = root
        root.withdraw()
        self.inicializando = True
        root.title("Sistema de Biometria — InsightFace + MediaPipe")
        root.geometry("1060x800")
        root.minsize(900, 720)
        root.configure(bg=self.BG)
        root.protocol("WM_DELETE_WINDOW", self.fechar)
        self.eventos = Queue()
        self.video = Queue(maxsize=1)
        self.server = None
        self.socket_api = None
        self.url = None
        self.token = secrets.token_urlsafe(32)
        self.camera = None
        self.preview_id = None
        self.sequencia_video = -1
        self.malha_em_movimento = MalhaEmMovimento()
        self.leitura_visual = LeituraVisual()
        self.geracao = 0
        self.ocupado = False
        self.pronto = False
        self.info = {"usuarios": [], "pasta": str(BASE / "dataset"), "avisos": []}
        self.conexoes = []
        self.frame = self.analise = None
        self.fotos = []
        self.ultima_captura = 0
        self.status = tk.StringVar(value="Preparando API local…")
        topo = tk.Frame(root, bg=self.BG)
        topo.pack(fill="x", padx=24, pady=(18, 8))
        self.rotulo(topo, "Sistema de Biometria", 22).pack(side="left")
        self.botao(topo, "Menu", self.menu, "#424242").pack(side="right", padx=5)
        self.botao(topo, "Preparar / tentar novamente", self.preparar, "#1976D2").pack(side="right", padx=5)
        self.conteudo = tk.Frame(root, bg=self.BG)
        self.conteudo.pack(fill="both", expand=True, padx=24, pady=8)
        tk.Label(root, textvariable=self.status, bg=self.BG, fg="#00BCD4", wraplength=990,
                 font=("Segoe UI", 10)).pack(fill="x", padx=20, pady=(5, 8))
        self.rotulo(root, "Processamento e fotos locais • Modelo de pesquisa • Reconhecimento não comprova presença real",
                    9, self.MUTED).pack(pady=(0, 10))
        self.menu()
        root.after(80, self.processar_eventos)
        root.after(150, self.preparar)

    def rotulo(self, pai, texto, tamanho=11, cor="white"):
        return tk.Label(pai, text=texto, bg=pai.cget("bg"), fg=cor,
                        font=("Segoe UI", tamanho), wraplength=920, justify="left")

    def botao(self, pai, texto, comando, cor="#2196F3"):
        return tk.Button(pai, text=texto, command=comando, bg=cor, fg="white", bd=0,
                         activebackground="#1976D2", activeforeground="white", cursor="hand2",
                         font=("Segoe UI", 11, "bold"), padx=14, pady=10)

    def bloquear(self, bloqueado, pai=None):
        for widget in (pai or self.root).winfo_children():
            if isinstance(widget, tk.Button):
                widget.configure(state="disabled" if bloqueado else "normal")
            self.bloquear(bloqueado, widget)

    def tarefa(self, trabalho, concluido, mensagem):
        if self.ocupado:
            return
        self.ocupado = True
        self.bloquear(True)
        self.status.set(mensagem)

        def executar():
            try:
                self.eventos.put((concluido, trabalho(), None))
            except Exception as erro:
                traceback.print_exc()
                self.eventos.put((None, None, str(erro)))

        threading.Thread(target=executar, daemon=True).start()

    def chamar(self, metodo, rota, sessao=None, **kwargs):
        if not self.url:
            raise RuntimeError("A API local ainda não iniciou. Clique em Preparar / tentar novamente.")
        if sessao is None:
            with requests.Session() as nova:
                nova.trust_env = False
                return self.chamar(metodo, rota, sessao=nova, **kwargs)
        try:
            resposta = sessao.request(metodo, self.url + rota,
                                       headers={"X-Biometria-Token": self.token},
                                       timeout=(5, 900), **kwargs)
            if not resposta.ok:
                try:
                    detalhe = resposta.json().get("detail", resposta.reason)
                except ValueError:
                    detalhe = resposta.reason
                raise RuntimeError(str(detalhe))
            return resposta
        except requests.Timeout as erro:
            raise RuntimeError("Tempo de espera esgotado. Confira o download dos modelos no terminal e tente novamente.") from erro
        except requests.RequestException as erro:
            raise RuntimeError("Não foi possível acessar a API local. Clique em Preparar / tentar novamente.") from erro

    def iniciar_api(self):
        if self.server and self.server.started:
            return
        faltando = [nome for nome in ("fastapi", "uvicorn", "multipart", "insightface", "mediapipe", "onnxruntime")
                    if importlib.util.find_spec(nome) is None]
        if faltando:
            raise RuntimeError("Dependências ausentes: " + ", ".join(faltando)
                               + f'\nInstale no terminal:\n& "{sys.executable}" -m pip install -r "{BASE / "requirements.txt"}"')
        import uvicorn
        from api_biometria import criar_api

        self.socket_api = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_api.bind(("127.0.0.1", 0))
        self.socket_api.listen(64)
        self.url = f"http://127.0.0.1:{self.socket_api.getsockname()[1]}"
        # pythonw/executáveis gráficos podem não ter stdout/stderr. O formatter
        # padrão do Uvicorn exige um terminal; o logging padrão do Python não.
        self.server = uvicorn.Server(uvicorn.Config(criar_api(token=self.token),
                                                    log_level="warning", log_config=None))
        thread = threading.Thread(target=self.server.run, kwargs={"sockets": [self.socket_api]}, daemon=True)
        thread.start()
        limite = time.monotonic() + 15
        while not self.server.started:
            if not thread.is_alive() or time.monotonic() > limite:
                raise RuntimeError("Não foi possível iniciar a API local.")
            time.sleep(0.05)
        print(f"API local iniciada em {self.url}. Encerrará junto com o aplicativo.")

    def preparar(self):
        if self.ocupado:
            return
        self.parar_camera()

        def trabalho():
            self.iniciar_api()
            return self.chamar("POST", "/preparar").json()

        def concluido(dados):
            self.pronto = True
            self.info = dados
            self.conexoes = dados["conexoes"]
            self.menu()
            self.status.set(f"Motor pronto em CPU. {len(dados['usuarios'])} usuário(s). "
                            f"{len(dados['avisos'])} aviso(s) na base.")
            self.root.deiconify()
            self.inicializando = False
            if dados["avisos"]:
                messagebox.showwarning("Fotos ignoradas", "\n".join(dados["avisos"][:12]))

        self.tarefa(trabalho, concluido,
                    "Preparando InsightFace + MediaPipe… No primeiro uso os modelos são baixados; acompanhe o terminal.")

    def exigir_pronto(self):
        if not self.pronto:
            messagebox.showinfo("Motor facial", "Aguarde a preparação ou clique em Preparar / tentar novamente.")
        return self.pronto

    def parar_camera(self):
        self.geracao += 1
        if self.preview_id is not None:
            self.root.after_cancel(self.preview_id)
            self.preview_id = None
        if self.camera:
            self.camera.parar.set()
            self.camera = None
        self.frame = self.analise = None
        self.malha_em_movimento = MalhaEmMovimento()
        self.leitura_visual = LeituraVisual()

    def limpar(self):
        self.parar_camera()
        for widget in self.conteudo.winfo_children():
            widget.destroy()

    def menu(self):
        if self.ocupado:
            return
        self.limpar()
        card = tk.Frame(self.conteudo, bg=self.CARD, padx=35, pady=24)
        card.pack(expand=True)
        self.rotulo(card, "Autenticação e geometria facial", 18).pack(pady=(0, 6))
        self.rotulo(card, "InsightFace + MediaPipe • API local", 11, self.MUTED).pack(pady=(0, 18))
        for texto, comando, cor in [
            ("Inscrever novo usuário", self.cadastro, "#388E3C"),
            ("Verificar acesso", self.verificacao, "#1976D2"),
            ("Análise de simetria facial", self.tela_simetria, "#008C9E"),
            ("Gerenciar usuários", self.usuarios, "#7B1FA2"),
            ("Armazenamento", self.armazenamento, "#B36B00"),
        ]:
            self.botao(card, texto, comando, cor).pack(fill="x", pady=6)
        self.rotulo(card, f"{len(self.info['usuarios'])} usuário(s) • fotos no computador", 10, self.MUTED).pack(pady=(16, 0))

    def cadastro(self):
        if not self.exigir_pronto():
            return
        nome = simpledialog.askstring("Cadastro", "Nome completo do novo usuário:", parent=self.root)
        if not nome or not nome.strip():
            return
        self.nome = nome.strip()
        self.fotos = []
        self.ultima_captura = 0
        self.tela_camera(f"Cadastro: {self.nome}", False)
        self.instrucao.set("1/3 — Olhe de frente, com expressão neutra.")
        self.botao_captura = self.botao(self.acoes_camera, "Capturar foto 1/3", self.capturar, "#388E3C")
        self.botao_captura.pack(side="left", padx=6)

    def verificacao(self):
        if not self.exigir_pronto():
            return
        if not any(u["amostras_validas"] for u in self.info["usuarios"]):
            messagebox.showinfo("Base vazia", "Cadastre uma pessoa antes de verificar o acesso.")
            return
        self.tela_camera("Reconhecimento facial ao vivo", True)
        self.instrucao.set("O nome aparece quando a similaridade e a separação das outras identidades são suficientes.")

    def tela_camera(self, titulo, reconhecer):
        self.limpar()
        self.rotulo(self.conteudo, titulo, 18).pack(pady=8)
        self.instrucao = tk.StringVar()
        tk.Label(self.conteudo, textvariable=self.instrucao, bg=self.BG, fg="#00BCD4",
                 wraplength=930, font=("Segoe UI", 12)).pack(pady=8)
        self.painel_video = tk.Label(self.conteudo, text="Abrindo webcam e analisando o primeiro quadro…",
                                    bg=self.CARD, fg="white")
        self.painel_video.pack(fill="both", expand=True, pady=10)
        self.acoes_camera = tk.Frame(self.conteudo, bg=self.BG)
        self.acoes_camera.pack(pady=8)
        self.botao(self.acoes_camera, "Encerrar câmera", self.menu, "#424242").pack(side="right", padx=6)
        self.camera = Camera()
        camera, geracao = self.camera, self.geracao
        self.sequencia_video = -1
        self.reconhecer = reconhecer

        def analisar():
            ultima = -1
            # Uma chamada por vez, conexão reutilizada e sem fila de quadros.
            with requests.Session() as sessao:
                sessao.trust_env = False
                while not camera.parar.is_set():
                    if camera.erro:
                        self.publicar_video((geracao, None, None, camera.erro))
                        return
                    frame, sequencia = camera.foto()
                    if frame is None or sequencia == ultima:
                        camera.parar.wait(0.02)
                        continue
                    ultima = sequencia
                    h, w = frame.shape[:2]
                    if max(h, w) > 960:
                        frame = cv2.resize(frame, (round(w * 960 / max(h, w)), round(h * 960 / max(h, w))))
                    instante = time.monotonic()
                    try:
                        resultado = self.chamar("POST", f"/analisar?reconhecer={'true' if reconhecer else 'false'}",
                                                sessao=sessao,
                                                files={"arquivo": ("frame.jpg", jpeg(frame), "image/jpeg")}).json()
                        resultado["capturado_em"] = instante
                        if camera.parar.is_set():
                            return
                        self.publicar_video((geracao, frame, resultado, None))
                    except Exception as erro:
                        if camera.parar.is_set():
                            return
                        self.publicar_video((geracao, None, None, str(erro)))
                        camera.parar.wait(1)
                    camera.parar.wait(max(0.005, 0.15 - (time.monotonic() - instante)))

        threading.Thread(target=analisar, daemon=True).start()
        self.atualizar_video()

    def atualizar_video(self):
        self.preview_id = None
        inicio = time.monotonic()
        if self.camera is None:
            return
        if self.camera.erro:
            self.frame = self.analise = None
            self.status.set(self.camera.erro)
            return
        frame, sequencia = self.camera.foto()
        if frame is not None and sequencia != self.sequencia_video:
            self.sequencia_video = sequencia
            # O vídeo nunca aguarda HTTP ou inferência. O fluxo óptico serve
            # apenas para acompanhar os polígonos entre leituras do modelo.
            malhas = self.malha_em_movimento.acompanhar(frame, inicio)
            self.desenhar_malha(frame, malhas)
            faces_visiveis = self.leitura_visual.visiveis(inicio)
            if faces_visiveis:
                for indice, face in enumerate(faces_visiveis):
                    texto = (face["nome"] or "ACESSO NEGADO") if self.reconhecer else "Rosto detectado"
                    if not face["qualidade_ok"]:
                        texto = "Aproxime-se da camera"
                    # Identidade é uma leitura recente do modelo; não é propagada
                    # pelo fluxo óptico para outras posições ou outros rostos.
                    ascii_texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
                    cv2.putText(frame, "Leitura: " + ascii_texto, (12, 26 + 26 * indice), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, (70, 220, 90) if face["nome"] else (0, 180, 255), 2)
            self.mostrar_imagem(self.painel_video, frame, rapido=True)
        atraso = max(1, round(33 - (time.monotonic() - inicio) * 1000))
        self.preview_id = self.root.after(atraso, self.atualizar_video)

    def publicar_video(self, item):
        try:
            self.video.get_nowait()
        except Empty:
            pass
        try:
            self.video.put_nowait(item)
        except Full:
            pass

    def capturar(self):
        if len(self.fotos) == 3:
            self.salvar_cadastro()
            return
        if self.frame is None or self.analise is None:
            messagebox.showinfo("Aguarde", "Aguarde a primeira análise do rosto.")
            return
        faces = self.analise["faces"]
        if time.monotonic() - self.analise["capturado_em"] > 1:
            messagebox.showinfo("Aguarde", "Aguarde uma análise recente antes de salvar a foto.")
            return
        if (self.analise["capturado_em"] <= self.ultima_captura
                or time.monotonic() - self.ultima_captura < 1):
            messagebox.showinfo("Próxima posição", "Mude a posição e aguarde um novo quadro analisado antes de capturar.")
            return
        if len(faces) != 1 or not faces[0]["qualidade_ok"]:
            messagebox.showwarning("Posicionamento", "Mantenha apenas um rosto no quadro e aproxime-se da câmera.")
            return
        # Salva o quadro limpo correspondente à análise exibida, sem polígonos.
        self.fotos.append(self.frame.copy())
        self.ultima_captura = time.monotonic()
        if len(self.fotos) == 3:
            self.salvar_cadastro()
        else:
            instrucoes = ["", "2/3 — Vire levemente para a sua direita; mantenha o rosto visível.",
                          "3/3 — Vire levemente para a sua esquerda; mantenha o rosto visível."]
            self.instrucao.set(instrucoes[len(self.fotos)])
            self.botao_captura.configure(text=f"Capturar foto {len(self.fotos) + 1}/3")

    def salvar_cadastro(self):
        self.parar_camera()
        self.botao_captura.configure(text="Tentar salvar cadastro")
        fotos, nome = self.fotos[:], self.nome

        def trabalho():
            arquivos = [("fotos", (f"foto_{i}.jpg", jpeg(foto), "image/jpeg")) for i, foto in enumerate(fotos, 1)]
            self.chamar("POST", f"/usuarios/{quote(nome, safe='')}/cadastro", files=arquivos)
            return self.chamar("GET", "/estado").json()

        def concluido(dados):
            self.info = dados
            self.fotos.clear()
            self.menu()
            self.status.set(f"{nome} cadastrado com três fotos locais.")

        self.tarefa(trabalho, concluido, "Validando e salvando as três fotos…")

    def desenhar_malha(self, imagem, malhas):
        h, w = imagem.shape[:2]
        for malha in malhas:
            pontos = np.rint(malha).astype(np.int32)
            if self.conexoes:
                if not hasattr(self, "_indices_malha"):
                    self._indices_malha = np.asarray(sorted({tuple(sorted(p)) for p in self.conexoes}), dtype=np.intp)
                indices = self._indices_malha
                indices = indices[(indices < len(pontos)).all(axis=1)]
                linhas = pontos[indices].reshape(-1, 2, 1, 2)
                cv2.polylines(imagem, linhas, False, (210, 210, 0), 1, cv2.LINE_AA)
            else:
                # Compatibilidade com versões sem a tabela de conexões exportada.
                pontos_unicos = np.unique(pontos[:468], axis=0)
                subdivisao = cv2.Subdiv2D((0, 0, w, h))
                for x, y in pontos_unicos:
                    if 0 <= x < w and 0 <= y < h:
                        subdivisao.insert((float(x), float(y)))
                for triangulo in subdivisao.getTriangleList():
                    vertices = np.rint(triangulo).astype(np.int32).reshape(3, 2)
                    if ((vertices >= 0).all() and (vertices[:, 0] < w).all() and (vertices[:, 1] < h).all()):
                        cv2.polylines(imagem, [vertices], True, (210, 210, 0), 1, cv2.LINE_AA)
        return imagem

    def mostrar_imagem(self, painel, frame, tamanho=(850, 440), rapido=False):
        if rapido:
            h, w = frame.shape[:2]
            fator = min(tamanho[0] / w, tamanho[1] / h, 1)
            if fator < 1:
                frame = cv2.resize(frame, (round(w * fator), round(h * fator)), interpolation=cv2.INTER_AREA)
        imagem = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not rapido:
            imagem.thumbnail(tamanho, Image.Resampling.LANCZOS)
        anterior = getattr(painel, "image", None)
        if rapido and anterior is not None and (anterior.width(), anterior.height()) == imagem.size:
            anterior.paste(imagem)
            return
        foto = ImageTk.PhotoImage(imagem)
        painel.configure(image=foto, text="")
        painel.image = foto

    def usuarios(self):
        if not self.exigir_pronto():
            return
        self.limpar()
        self.rotulo(self.conteudo, "Gerenciamento de usuários", 18).pack(pady=15)
        self.lista = tk.Listbox(self.conteudo, bg=self.CARD, fg="white", selectbackground="#1976D2",
                               font=("Segoe UI", 13), bd=0, height=10)
        self.lista.pack(fill="both", expand=True, padx=60, pady=15)
        self.nomes_lista = [u["nome"] for u in self.info["usuarios"]]
        for usuario in self.info["usuarios"]:
            self.lista.insert("end", f"{usuario['nome']} — {usuario['fotos']} foto(s), {usuario['amostras_validas']} amostra(s) válida(s)")
        self.botao(self.conteudo, "Remover usuário selecionado", self.remover_usuario, "#C62828").pack(pady=15)

    def selecionado(self):
        indices = self.lista.curselection()
        if not indices:
            messagebox.showinfo("Selecione", "Selecione um usuário na lista.")
            return None
        return self.nomes_lista[indices[0]]

    def remover_usuario(self):
        nome = self.selecionado()
        if not nome or not messagebox.askyesno("Remover usuário", f"Excluir o cadastro e todas as fotos locais de {nome}?"):
            return

        def trabalho():
            self.chamar("DELETE", f"/usuarios/{quote(nome, safe='')}")
            return self.chamar("GET", "/estado").json()

        def concluido(dados):
            self.info = dados
            self.usuarios()
            self.status.set("Cadastro e referências removidos.")

        self.tarefa(trabalho, concluido, "Removendo cadastro local…")

    def armazenamento(self):
        if not self.exigir_pronto():
            return
        self.limpar()
        self.rotulo(self.conteudo, "Armazenamento local", 18).pack(pady=25)
        self.rotulo(self.conteudo, self.info["pasta"], 12, "#4CAF50").pack(pady=20)
        self.rotulo(self.conteudo, "Trocar a pasta seleciona outra base; as fotos existentes não são movidas.",
                    cor=self.MUTED).pack(pady=15)
        self.botao(self.conteudo, "Escolher pasta de fotos", self.escolher_pasta).pack(pady=20)

    def escolher_pasta(self):
        pasta = filedialog.askdirectory(title="Selecione a pasta dos cadastros locais")
        if not pasta:
            return

        def concluido(dados):
            self.info = dados
            self.armazenamento()
            self.status.set(f"Base selecionada. {len(dados['avisos'])} aviso(s).")
            if dados["avisos"]:
                messagebox.showwarning("Fotos ignoradas", "\n".join(dados["avisos"][:12]))

        self.tarefa(lambda: self.chamar("PUT", "/armazenamento", json={"pasta_armazenamento": pasta}).json(),
                    concluido, "Selecionando pasta e extraindo referências faciais das fotos…")

    def tela_simetria(self):
        if not self.exigir_pronto():
            return
        self.limpar()
        self.rotulo(self.conteudo, "Análise de simetria facial", 18).pack(pady=15)
        self.rotulo(self.conteudo, "Use uma foto frontal, sem obstruções e com expressão neutra.",
                    cor=self.MUTED).pack(pady=8)
        self.botao(self.conteudo, "Selecionar foto avulsa", self.foto_avulsa, "#008C9E").pack(pady=10)
        self.lista = tk.Listbox(self.conteudo, bg=self.CARD, fg="white", selectbackground="#008C9E",
                               font=("Segoe UI", 13), height=8, bd=0)
        self.lista.pack(fill="both", expand=True, padx=60, pady=15)
        self.nomes_lista = [u["nome"] for u in self.info["usuarios"]]
        for nome in self.nomes_lista:
            self.lista.insert("end", nome)
        self.botao(self.conteudo, "Analisar foto frontal do usuário", self.foto_cadastrada, "#008C9E").pack(pady=12)

    def foto_avulsa(self):
        caminho = filedialog.askopenfilename(filetypes=[("Imagens", "*.jpg *.jpeg *.png *.webp *.bmp")])
        if not caminho:
            return

        def trabalho():
            with Image.open(caminho) as original:
                if original.width * original.height > 20_000_000:
                    raise ValueError("Selecione uma foto de até 20 milhões de pixels.")
                imagem = ImageOps.exif_transpose(original).convert("RGB")
                imagem.thumbnail((960, 960))
            frame = cv2.cvtColor(np.array(imagem), cv2.COLOR_RGB2BGR)
            return self.analisar_simetria(frame)

        self.tarefa(trabalho, self.resultado_simetria, "Localizando pontos e comparando os dois lados da foto…")

    def foto_cadastrada(self):
        nome = self.selecionado()
        if not nome:
            return

        def trabalho():
            resposta = self.chamar("GET", f"/usuarios/{quote(nome, safe='')}/foto")
            frame = cv2.imdecode(np.frombuffer(resposta.content, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Foto cadastrada inválida.")
            return self.analisar_simetria(frame)

        self.tarefa(trabalho, self.resultado_simetria, "Analisando a foto frontal cadastrada…")

    def analisar_simetria(self, frame):
        resultado = self.chamar("POST", "/simetria", files={"arquivo": ("foto.jpg", jpeg(frame), "image/jpeg")}).json()
        return frame, resultado

    def resultado_simetria(self, dados):
        frame, resultado = dados
        self.limpar()
        self.rotulo(self.conteudo, f"Simetria geométrica estimada: {resultado['pontuacao']:.1f}/100", 20).pack(pady=12)
        colunas = tk.Frame(self.conteudo, bg=self.BG)
        colunas.pack(fill="both", expand=True)
        imagem = tk.Label(colunas, bg=self.CARD)
        imagem.pack(side="left", expand=True, fill="both", padx=10)
        anotada = self.desenhar_malha(frame.copy(), resultado["malhas"])
        eixo = np.rint(resultado["eixo"]).astype(int)
        cv2.line(anotada, tuple(eixo[0]), tuple(eixo[1]), (0, 255, 255), 2, cv2.LINE_AA)
        self.mostrar_imagem(imagem, anotada, (500, 400))
        detalhes = tk.Frame(colunas, bg=self.BG)
        detalhes.pack(side="right", padx=15)
        for regiao, valores in resultado["regioes"].items():
            self.rotulo(detalhes, f"{regiao}: {valores['pontuacao']:.1f}/100", 13).pack(anchor="w", pady=(12, 3))
            self.rotulo(detalhes, f"Desvio: {valores['desvio_percentual']:.2f}% da largura facial",
                        10, self.MUTED).pack(anchor="w")
        self.rotulo(self.conteudo, "Pontuação = 100 × (1 − erro médio dos pares espelhados / largura facial).",
                    10, self.MUTED).pack(pady=12)
        self.rotulo(self.conteudo, resultado["nota"], 10, "#FFB74D").pack(pady=8)
        self.botao(self.conteudo, "Analisar outra foto", self.tela_simetria, "#008C9E").pack(pady=12)
        self.status.set("Análise concluída. Fotos avulsas não são incluídas nos cadastros.")

    def processar_eventos(self):
        try:
            callback, resultado, erro = self.eventos.get_nowait()
        except Empty:
            pass
        else:
            self.ocupado = False
            self.bloquear(False)
            if erro:
                self.status.set("A operação falhou. Confira o terminal e tente novamente.")
                if self.inicializando:
                    if messagebox.askretrycancel("Falha ao iniciar", erro, parent=self.root):
                        self.root.after(100, self.preparar)
                    else:
                        self.fechar()
                        return
                else:
                    messagebox.showerror("Sistema de Biometria", erro)
            else:
                try:
                    callback(resultado)
                except Exception:
                    traceback.print_exc()
                    messagebox.showerror("Interface", "Não foi possível apresentar o resultado. Confira o terminal.")
                    if self.inicializando:
                        self.fechar()
                        return
        try:
            geracao, frame, analise, erro = self.video.get_nowait()
        except Empty:
            pass
        else:
            if geracao == self.geracao:
                if erro:
                    self.frame = self.analise = None
                    self.malha_em_movimento = MalhaEmMovimento()
                    self.leitura_visual = LeituraVisual()
                    self.status.set(erro)
                else:
                    self.frame, self.analise = frame, analise
                    recebido_em = time.monotonic()
                    self.leitura_visual.atualizar(analise["faces"], recebido_em)
                    self.malha_em_movimento.atualizar(
                        frame, analise["malhas"], analise["capturado_em"],
                        recebido_em=recebido_em, validade=self.leitura_visual.validade)
                    nomes = []
                    for face in self.leitura_visual.visiveis(recebido_em):
                        nome = face["nome"]
                        texto = (nome or "ACESSO NEGADO") if self.reconhecer else "Rosto detectado"
                        if not face["qualidade_ok"]:
                            texto = "Aproxime-se da camera"
                        nomes.append(texto + (f" · cosseno {face['similaridade']:.3f}" if face["similaridade"] is not None else ""))
                    self.status.set("Última análise: " + " | ".join(nomes) if nomes else "Nenhum rosto detectado. Posicione-se diante da câmera.")
        self.root.after(30, self.processar_eventos)

    def fechar(self):
        self.parar_camera()
        if self.server:
            self.server.should_exit = True
        self.root.destroy()


if __name__ == "__main__":
    janela = tk.Tk()
    BiometriaApp(janela)
    janela.mainloop()
