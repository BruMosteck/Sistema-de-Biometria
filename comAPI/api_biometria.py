"""API local e motor facial. Iniciada automaticamente por app_biometria.py."""

from io import BytesIO
import base64
import json
import logging
import os
from pathlib import Path
import re
import shutil
import threading
from types import SimpleNamespace
from typing import Literal
import uuid
import warnings

import cv2
from fastapi import FastAPI, HTTPException, UploadFile
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel

BASE = Path(__file__).resolve().parent
MODELO = "buffalo_l"
LIMIAR = 0.50  # Similaridade de cosseno, não porcentagem de certeza.
MARGEM = 0.05  # Diferença mínima entre as duas melhores identidades.
MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 20_000_000
PARES = {
    "Olhos": [(33, 263), (133, 362), (159, 386), (145, 374)],
    "Sobrancelhas": [(70, 300), (105, 334), (107, 336)],
    "Boca": [(61, 291), (78, 308), (82, 312)],
    "Mandíbula": [(234, 454), (93, 323), (172, 397)],
}
OVAL_FACIAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
              397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
              172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


def nome_seguro(nome):
    nome = nome.strip()
    reservados = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                  *(f"LPT{i}" for i in range(1, 10))}
    if (not nome or len(nome) > 80 or nome in {".", ".."}
            or re.search(r'[<>:"/\\|?*\x00-\x1f]', nome) or nome.endswith((".", " "))
            or nome.split(".")[0].upper() in reservados):
        raise HTTPException(400, "Use um nome de até 80 caracteres, sem símbolos de caminho ou nomes reservados.")
    return nome


def ler_imagem(arquivo):
    try:
        dados = arquivo.file.read(MAX_BYTES + 1)
    finally:
        arquivo.file.close()
    if len(dados) > MAX_BYTES:
        raise HTTPException(413, "Imagem maior que 8 MiB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(dados)) as original:
                if original.format not in {"JPEG", "PNG", "WEBP", "BMP"}:
                    raise HTTPException(415, "Use JPEG, PNG, WEBP ou BMP.")
                if original.width * original.height > MAX_PIXELS:
                    raise HTTPException(413, "Imagem maior que 20 milhões de pixels.")
                imagem = ImageOps.exif_transpose(original).convert("RGB")
                imagem.thumbnail((960, 960))
                return cv2.cvtColor(np.array(imagem), cv2.COLOR_RGB2BGR)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as erro:
        raise HTTPException(413, "Dimensões da imagem excessivas.") from erro
    except (UnidentifiedImageError, OSError, ValueError) as erro:
        raise HTTPException(400, "Arquivo de imagem inválido ou vazio.") from erro


def vetor_normalizado(vetor):
    vetor = np.asarray(vetor, dtype=np.float32)
    norma = np.linalg.norm(vetor)
    if vetor.ndim != 1 or not np.isfinite(vetor).all() or norma < 1e-8:
        raise ValueError("Vetor facial inválido.")
    return vetor / norma


def comparar(vetor, referencias):
    # Mantém cada ângulo de cadastro como referência, em vez de fazer a média.
    placar = sorted(((nome, max(float(vetor @ ref) for ref in amostras))
                     for nome, amostras in referencias.items() if amostras),
                    key=lambda item: item[1], reverse=True)
    if not placar:
        return None, None
    nome, valor = placar[0]
    ambiguo = len(placar) > 1 and valor - placar[1][1] < MARGEM
    return (nome if valor >= LIMIAR and not ambiguo else None), round(valor, 4)


def calcular_simetria(pontos):
    """Erro de pares espelhados, normalizado pela largura facial; métrica didática."""
    pontos = np.asarray(pontos, dtype=float)
    if pontos.ndim != 2 or pontos.shape[0] < 455 or pontos.shape[1] != 2 or not np.isfinite(pontos).all():
        raise HTTPException(422, "Pontos faciais insuficientes ou inválidos.")
    olho_a = pontos[[33, 133]].mean(axis=0)
    olho_b = pontos[[263, 362]].mean(axis=0)
    eixo_x = olho_b - olho_a
    distancia_olhos = np.linalg.norm(eixo_x)
    if distancia_olhos < 10:
        raise HTTPException(422, "Aproxime o rosto da câmera.")
    eixo_x /= distancia_olhos
    eixo_y = np.array([-eixo_x[1], eixo_x[0]])
    origem = (pontos[10] + pontos[152]) / 2
    alinhados = (pontos - origem) @ np.stack((eixo_x, eixo_y), axis=1)
    largura = abs(alinhados[454, 0] - alinhados[234, 0])
    if largura < 20:
        raise HTTPException(422, "Rosto pequeno demais para analisar simetria.")
    regioes, erros = {}, []
    for regiao, pares in PARES.items():
        valores = [float(np.linalg.norm(alinhados[a] * [-1, 1] - alinhados[b]) / largura)
                   for a, b in pares]
        erro = float(np.mean(valores))
        erros.extend(valores)
        regioes[regiao] = {"pontuacao": round(max(0, 100 * (1 - erro)), 1),
                           "desvio_percentual": round(100 * erro, 2)}
    return {"pontuacao": round(max(0, 100 * (1 - float(np.mean(erros)))), 1),
            "regioes": regioes,
            "eixo": [pontos[10].tolist(), pontos[152].tolist()],
            "nota": "Estimativa geométrica da foto, sensível a pose e expressão; não é uma avaliação clínica."}


def gerar_rosto_simetrico(imagem, pontos, lado):
    """Alinha um retrato e espelha a metade escolhida sem gerar novos traços."""
    if lado not in {"direita", "esquerda"}:
        raise HTTPException(422, "Escolha o lado direito ou esquerdo do rosto.")
    pontos = np.asarray(pontos, dtype=np.float64)
    if (pontos.ndim != 2 or pontos.shape[0] < 455 or pontos.shape[1] != 2
            or not np.isfinite(pontos).all()):
        raise HTTPException(422, "Pontos faciais insuficientes para gerar a imagem.")
    olho_direito = pontos[[33, 133]].mean(axis=0)
    olho_esquerdo = pontos[[263, 362]].mean(axis=0)
    horizontal = olho_esquerdo - olho_direito
    distancia = np.linalg.norm(horizontal)
    if distancia < 10:
        raise HTTPException(422, "O rosto está pequeno demais para gerar a simetria.")
    horizontal /= distancia
    vertical = np.array([-horizontal[1], horizontal[0]])
    centro = (pontos[10] + pontos[152]) / 2
    oval = (pontos[OVAL_FACIAL] - centro) @ np.stack([horizontal, vertical], axis=1)
    altura_rosto = float(np.ptp(oval[:, 1]))
    meia_largura = int(np.ceil(np.max(np.abs(oval[:, 0])) * 1.25))
    topo = int(np.floor(oval[:, 1].min() - altura_rosto * 0.25))
    fundo = int(np.ceil(oval[:, 1].max() + altura_rosto * 0.12))
    if altura_rosto < 20 or meia_largura < 10:
        raise HTTPException(422, "Não foi possível obter o contorno do rosto.")
    # Limita a área gerada mesmo se uma medição do modelo vier distorcida.
    h, w = imagem.shape[:2]
    if meia_largura * 2 + 1 > w * 3 or fundo - topo + 1 > h * 3:
        raise HTTPException(422, "Contorno facial inválido. Tente outra foto frontal.")
    matriz = np.array([
        [*horizontal, meia_largura - float(horizontal @ centro)],
        [*vertical, -topo - float(vertical @ centro)],
    ], dtype=np.float64)
    original = cv2.warpAffine(imagem, matriz, (2 * meia_largura + 1, fundo - topo + 1),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    simetrica = original.copy()
    # Foto não espelhada: o lado direito da pessoa aparece à esquerda da foto.
    # Largura ímpar mantém uma coluna central única e dois lados de mesmo tamanho.
    if lado == "direita":
        simetrica[:, meia_largura + 1:] = original[:, :meia_largura][:, ::-1]
    else:
        simetrica[:, :meia_largura] = original[:, meia_largura + 1:][:, ::-1]
    return original, simetrica


def png_base64(imagem):
    sucesso, dados = cv2.imencode(".png", imagem)
    if not sucesso:
        raise RuntimeError("Não foi possível codificar a imagem gerada.")
    return base64.b64encode(dados).decode("ascii")


class MotorFacial:
    def __init__(self, base=BASE):
        self.base = Path(base)
        self.config = self.base / "config_biometria.json"
        self.pasta = self.base / "dataset"
        self.referencias = {}
        self.avisos = []
        self.facial = self.malha = self.mp = None
        self.conexoes = []
        self.lock = threading.RLock()
        if self.config.exists():
            try:
                caminho = json.loads(self.config.read_text(encoding="utf-8"))["pasta_armazenamento"]
                self.pasta = (self.base / caminho).resolve()
            except (OSError, ValueError, KeyError, TypeError):
                self.avisos.append("Configuração inválida; usando a pasta dataset padrão.")

    def pasta_usuario(self, nome):
        caminho = self.pasta / nome_seguro(nome)
        if caminho.is_symlink() or caminho.resolve().parent != self.pasta.resolve():
            raise HTTPException(400, "Diretório de usuário inválido.")
        return caminho

    def usuarios(self):
        if not self.pasta.exists():
            return []
        return [{"nome": p.name, "fotos": len(self.fotos(p)),
                 "amostras_validas": len(self.referencias.get(p.name, []))}
                for p in sorted(self.pasta.iterdir(), key=lambda p: p.name.casefold())
                if p.is_dir() and not p.is_symlink() and not p.name.startswith(".")]

    @staticmethod
    def fotos(pasta):
        return sorted(p for p in pasta.iterdir() if p.is_file() and not p.is_symlink()
                      and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"})

    def preparar(self):
        if self.facial is not None and self.malha is not None:
            return
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision
        from insightface.app import FaceAnalysis
        import onnxruntime as ort
        import requests

        modelos = self.base / "modelos"
        modelos.mkdir(parents=True, exist_ok=True)
        tarefa = modelos / "face_landmarker.task"
        if not tarefa.exists():
            url = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                   "face_landmarker/float16/1/face_landmarker.task")
            temporario = tarefa.with_suffix(".part")
            try:
                with requests.get(url, stream=True, timeout=(15, 120)) as resposta:
                    resposta.raise_for_status()
                    with temporario.open("wb") as destino:
                        for bloco in resposta.iter_content(1024 * 1024):
                            destino.write(bloco)
                temporario.replace(tarefa)
            finally:
                temporario.unlink(missing_ok=True)
        # Evita vários pools ocupando todos os núcleos enquanto Tk e MediaPipe
        # também precisam da CPU. Preserva o modelo e a resolução do detector.
        opcoes = ort.SessionOptions()
        opcoes.intra_op_num_threads = min(4, max(1, (os.cpu_count() or 2) // 2))
        opcoes.inter_op_num_threads = 1
        opcoes.add_session_config_entry("session.intra_op.allow_spinning", "0")
        opcoes.add_session_config_entry("session.inter_op.allow_spinning", "0")
        facial = FaceAnalysis(name=MODELO, root=str(modelos),
                              allowed_modules=["detection", "recognition"],
                              providers=["CPUExecutionProvider"], sess_options=opcoes)
        facial.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.60)
        malha = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(tarefa)),
            running_mode=vision.RunningMode.IMAGE, num_faces=4,
            output_facial_transformation_matrixes=True, output_face_blendshapes=True,
        ))
        conexoes = getattr(vision, "FaceLandmarksConnections", None)
        if conexoes:
            for atributo in ("FACE_LANDMARKS_TESSELATION", "FACE_LANDMARKS_TESSELLATION"):
                if hasattr(conexoes, atributo):
                    self.conexoes = sorted({tuple(sorted((c.start, c.end)))
                                            for c in getattr(conexoes, atributo)})
                    break
        self.mp, self.facial, self.malha = mp, facial, malha
        self.pasta.mkdir(parents=True, exist_ok=True)
        self.reindexar()

    def reindexar(self):
        referencias, avisos = {}, []
        for usuario in self.usuarios():
            nome = usuario["nome"]
            amostras = []
            for foto in self.fotos(self.pasta_usuario(nome)):
                try:
                    imagem = cv2.imdecode(np.frombuffer(foto.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
                    if imagem is None:
                        raise ValueError("Imagem ilegível")
                    h, w = imagem.shape[:2]
                    if max(h, w) > 960:
                        imagem = cv2.resize(imagem, (round(w * 960 / max(h, w)), round(h * 960 / max(h, w))))
                    faces = self.facial.get(imagem)
                    if len(faces) != 1:
                        raise ValueError("Foto precisa conter exatamente um rosto")
                    amostras.append(vetor_normalizado(faces[0].normed_embedding))
                except Exception as erro:
                    avisos.append(f"{nome}/{foto.name}: {erro}")
            if amostras:
                referencias[nome] = amostras
        self.referencias = referencias
        self.avisos = avisos

    def pontos(self, imagem):
        rgb = np.ascontiguousarray(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
        resultado = self.malha.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb))
        h, w = imagem.shape[:2]
        malhas = [[[round(p.x * w, 2), round(p.y * h, 2)] for p in face]
                  for face in resultado.face_landmarks]
        return resultado, malhas

    def analisar(self, imagem, reconhecer):
        if reconhecer:
            faces = self.facial.get(imagem)
        else:
            # Na prévia de cadastro basta localizar o rosto. As características
            # são extraídas e validadas nas três fotos ao salvar o cadastro.
            caixas, _ = self.facial.det_model.detect(imagem, max_num=0, metric="default")
            faces = [SimpleNamespace(bbox=caixa[:4], det_score=caixa[4]) for caixa in caixas]
        _, malhas = self.pontos(imagem)
        saida = []
        h, w = imagem.shape[:2]
        for face in faces:
            x1, y1, x2, y2 = np.rint(face.bbox).astype(int)
            caixa = [int(np.clip(x1, 0, w)), int(np.clip(y1, 0, h)),
                     int(np.clip(x2, 0, w)), int(np.clip(y2, 0, h))]
            adequado = x2 - x1 >= 80 and y2 - y1 >= 80 and float(face.det_score) >= 0.65
            nome, similaridade = (None, None)
            if reconhecer and adequado:
                nome, similaridade = comparar(vetor_normalizado(face.normed_embedding), self.referencias)
            saida.append({"caixa": caixa, "nome": nome, "similaridade": similaridade,
                          "qualidade_ok": adequado})
        return {"faces": saida, "malhas": malhas, "largura": w, "altura": h}

    def cadastrar(self, nome, imagens):
        pasta = self.pasta_usuario(nome)
        if pasta.exists():
            raise HTTPException(409, "Esse usuário já existe. Escolha outro nome ou remova o cadastro anterior.")
        amostras = []
        for imagem in imagens:
            faces = self.facial.get(imagem)
            if len(faces) != 1:
                raise HTTPException(422, "Cada foto deve conter exatamente um rosto.")
            face = faces[0]
            x1, y1, x2, y2 = face.bbox
            if min(x2 - x1, y2 - y1) < 80 or float(face.det_score) < 0.65:
                raise HTTPException(422, "Rosto pequeno ou pouco nítido. Refaça as capturas mais perto da câmera.")
            amostras.append(vetor_normalizado(face.normed_embedding))
        temporario = self.pasta / f".cadastro-{uuid.uuid4().hex}"
        temporario.mkdir()
        try:
            for i, imagem in enumerate(imagens, 1):
                sucesso, dados = cv2.imencode(".jpg", imagem, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not sucesso:
                    raise OSError("Falha ao codificar foto")
                (temporario / f"foto_{i}.jpg").write_bytes(dados.tobytes())
            temporario.rename(pasta)
        finally:
            if temporario.exists():
                shutil.rmtree(temporario)
        self.referencias[nome] = amostras

    def simetria(self, imagem):
        resultado, malhas = self.pontos(imagem)
        if len(malhas) != 1:
            raise HTTPException(422, "Use uma foto com exatamente um rosto frontal e sem obstruções.")
        if not resultado.facial_transformation_matrixes:
            raise HTTPException(422, "Não foi possível estimar a pose. Tente outra foto frontal.")
        matriz = np.asarray(resultado.facial_transformation_matrixes[0])[:3, :3]
        u, _, vt = np.linalg.svd(matriz)
        rotacao = u @ vt
        yaw = float(np.degrees(np.arctan2(rotacao[0, 2], np.hypot(rotacao[0, 0], rotacao[0, 1]))))
        pitch = float(np.degrees(np.arctan2(-rotacao[1, 2], rotacao[2, 2])))
        if abs(yaw) > 15 or abs(pitch) > 20:
            raise HTTPException(422, "Olhe de frente para a câmera, sem inclinar a cabeça para cima ou para os lados.")
        expressoes = {p.category_name: p.score for p in resultado.face_blendshapes[0]}
        if max(expressoes.get("jawOpen", 0), expressoes.get("mouthSmileLeft", 0),
               expressoes.get("mouthSmileRight", 0)) > 0.45:
            raise HTTPException(422, "Use uma expressão neutra, com a boca relaxada.")
        return {**calcular_simetria(malhas[0]), "malhas": malhas,
                "pose": {"yaw": round(yaw, 1), "pitch": round(pitch, 1)}}


class Configuracao(BaseModel):
    pasta_armazenamento: str


def criar_api(motor=None, token=None):
    from fastapi import Request
    from fastapi.responses import JSONResponse

    motor = motor or MotorFacial()
    app = FastAPI(title="Biometria ESEG — API local", version="0.2.0")

    @app.middleware("http")
    async def autenticar(request: Request, call_next):
        if token and request.headers.get("X-Biometria-Token") != token:
            return JSONResponse({"detail": "Cliente local não autorizado."}, status_code=401)
        return await call_next(request)

    @app.exception_handler(Exception)
    async def falha_motor(request: Request, erro: Exception):
        logging.exception("Falha na API facial", exc_info=erro)
        return JSONResponse({"detail": "Falha no motor facial. Confira o terminal. "
                             "Verifique as dependências, a conexão para baixar os modelos e tente novamente."},
                            status_code=503)

    @app.get("/estado")
    def estado():
        with motor.lock:
            return {"pronto": motor.facial is not None and motor.malha is not None,
                    "pasta": str(motor.pasta), "usuarios": motor.usuarios(), "avisos": motor.avisos}

    @app.post("/preparar")
    def preparar():
        with motor.lock:
            motor.preparar()
            return {**estado(), "conexoes": motor.conexoes}

    @app.post("/analisar")
    def analisar(arquivo: UploadFile, reconhecer: bool = True):
        imagem = ler_imagem(arquivo)
        with motor.lock:
            motor.preparar()
            return motor.analisar(imagem, reconhecer)

    @app.post("/usuarios/{nome}/cadastro", status_code=201)
    def cadastrar(nome: str, fotos: list[UploadFile]):
        nome = nome_seguro(nome)
        if len(fotos) != 3:
            raise HTTPException(422, "Envie exatamente três fotos: frente, direita e esquerda.")
        imagens = [ler_imagem(foto) for foto in fotos]
        with motor.lock:
            motor.preparar()
            motor.cadastrar(nome, imagens)
            return {"nome": nome, "fotos": 3}

    @app.delete("/usuarios/{nome}")
    def remover(nome: str):
        with motor.lock:
            pasta = motor.pasta_usuario(nome)
            if not pasta.is_dir():
                raise HTTPException(404, "Usuário não encontrado.")
            shutil.rmtree(pasta)
            motor.referencias.pop(nome, None)
            return {"removido": nome}

    @app.put("/armazenamento")
    def armazenamento(config: Configuracao):
        pasta = Path(config.pasta_armazenamento).expanduser()
        if not pasta.is_absolute():
            raise HTTPException(400, "Escolha uma pasta com caminho absoluto.")
        with motor.lock:
            pasta.mkdir(parents=True, exist_ok=True)
            anterior = motor.pasta
            try:
                motor.pasta = pasta.resolve()
                if motor.facial is not None:
                    motor.reindexar()
                temporario = motor.config.with_suffix(".tmp")
                temporario.write_text(json.dumps({"pasta_armazenamento": str(motor.pasta)}, ensure_ascii=False), encoding="utf-8")
                temporario.replace(motor.config)
            except Exception:
                motor.pasta = anterior
                if motor.facial is not None:
                    motor.reindexar()
                raise
            return estado()

    @app.post("/simetria")
    def simetria(arquivo: UploadFile):
        imagem = ler_imagem(arquivo)
        with motor.lock:
            motor.preparar()
            return motor.simetria(imagem)

    @app.post("/simetria/gerar")
    def gerar_simetria(arquivo: UploadFile, lado: Literal["direita", "esquerda"]):
        imagem = ler_imagem(arquivo)
        with motor.lock:
            motor.preparar()
            analise = motor.simetria(imagem)
            original, simetrica = gerar_rosto_simetrico(imagem, analise["malhas"][0], lado)
        return {"lado": lado, "original_png": png_base64(original),
                "simetrica_png": png_base64(simetrica),
                "nota": "As duas imagens usam o mesmo alinhamento e recorte. "
                        "A simulação espelha a metade escolhida, incluindo o entorno do rosto."}

    @app.get("/usuarios/{nome}/foto")
    def foto_usuario(nome: str):
        from fastapi.responses import Response

        with motor.lock:
            pasta = motor.pasta_usuario(nome)
            fotos = motor.fotos(pasta) if pasta.is_dir() else []
            if not fotos:
                raise HTTPException(404, "Nenhuma foto disponível para este usuário.")
            # A foto frontal é foto_1; normaliza formato e orientação na saída.
            with fotos[0].open("rb") as arquivo:
                from types import SimpleNamespace
                imagem = ler_imagem(SimpleNamespace(file=arquivo))
            sucesso, dados = cv2.imencode(".jpg", imagem)
            if not sucesso:
                raise HTTPException(422, "Foto inválida.")
            return Response(dados.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    return app
