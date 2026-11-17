"""Reconhecimento por pontos dominantes, sem modelos ou treinamento.

Gao, Y. (2003), Efficiently comparing face images using a modified
Hausdorff distance, doi:10.1049/ip-vis:20030805, seções 2–4, eqs. (4)–(5).
OpenCV executa operações de imagem/geometria; NumPy executa os cálculos.
"""

from pathlib import Path
from concurrent.futures import CancelledError
import json

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont


TAMANHO = (160, 200)  # largura, altura do sistema de coordenadas comum
OLHOS_DESTINO = np.array([[48., 52.], [112., 52.]])
LARGURAS_FAIXA = np.arange(0.5, 2.51, 0.5)
ARQUIVO_GEOMETRIA = "geometria_classica.json"
EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VERSAO_DESCRITOR = "m2hd-160x200-canny80-160-faixas0.5-2.5-v1"


def verificar_cancelamento(cancelar):
    if cancelar is not None and cancelar.is_set():
        raise CancelledError()


def _ler_geometria(arquivo):
    try:
        dados = json.loads(arquivo.read_text(encoding="utf-8"))
        return dados if isinstance(dados, dict) else {}
    except (OSError, ValueError):
        return {}


def _gravar_geometria(arquivo, dados):
    temporario = arquivo.with_suffix(".tmp")
    temporario.write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")
    temporario.replace(arquivo)


def ler_imagem(caminho):
    """BGR, incluindo nomes Unicode e orientação EXIF, sem localizar rostos."""
    with Image.open(caminho) as imagem:
        return cv2.cvtColor(np.array(ImageOps.exif_transpose(imagem).convert("RGB")),
                            cv2.COLOR_RGB2BGR)


def matriz_alinhamento(imagem, olhos):
    """Similaridade 2D determinada por dois olhos marcados, sem estimação de modelo."""
    pontos = np.asarray(olhos, dtype=np.float64)
    if pontos.shape != (2, 2) or not np.isfinite(pontos).all():
        raise ValueError("Marque os centros dos dois olhos.")
    pontos = pontos[np.argsort(pontos[:, 0])]
    altura, largura = imagem.shape[:2]
    if np.any(pontos < 0) or np.any(pontos >= [largura, altura]):
        raise ValueError("Os olhos devem estar dentro da imagem.")
    dx, dy = pontos[1] - pontos[0]
    norma2 = dx * dx + dy * dy
    if norma2 < 100:
        raise ValueError("Os olhos estão muito próximos. Use uma foto maior.")
    # [a b; -b a]: rotação e escala, sem cisalhamento ou reflexão.
    a, b = 64 * dx / norma2, 64 * dy / norma2
    linear = np.array([[a, b], [-b, a]])
    matriz = np.column_stack((linear, OLHOS_DESTINO[0] - linear @ pontos[0]))
    inversa = cv2.invertAffineTransform(matriz)
    cantos = np.array([[0., 0., 1.], [159., 0., 1.],
                       [159., 199., 1.], [0., 199., 1.]]) @ inversa.T
    if np.any(cantos < 0) or np.any(cantos > [largura - 1, altura - 1]):
        raise ValueError("O recorte sai da foto. Afaste-se da câmera e mantenha o rosto inteiro visível.")
    return matriz


def normalizar_rosto(imagem, olhos):
    """Alinha escala/posição pelos olhos e recorta sobrancelhas até o queixo."""
    return cv2.warpAffine(imagem, matriz_alinhamento(imagem, olhos), TAMANHO,
                          flags=cv2.INTER_LINEAR)


def afinar_bordas(bordas):
    """Afinamento binário Zhang–Suen; apenas vizinhanças e regras fixas."""
    mascara = np.pad((bordas > 0).astype(np.uint8), 1)
    while True:
        mudou = False
        for etapa in (0, 1):
            p = mascara[1:-1, 1:-1]
            vizinhos = [mascara[:-2, 1:-1], mascara[:-2, 2:],
                        mascara[1:-1, 2:], mascara[2:, 2:],
                        mascara[2:, 1:-1], mascara[2:, :-2],
                        mascara[1:-1, :-2], mascara[:-2, :-2]]
            quantidade = sum(vizinhos)
            transicoes = sum((vizinhos[k] == 0) & (vizinhos[(k + 1) % 8] == 1)
                             for k in range(8))
            n, _, e, _, s, _, w, _ = vizinhos
            if etapa == 0:
                condicao = (n * e * s == 0) & (e * s * w == 0)
            else:
                condicao = (n * e * w == 0) & (n * s * w == 0)
            apagar = ((p == 1) & (quantidade >= 2) & (quantidade <= 6)
                      & (transicoes == 1) & condicao)
            if apagar.any():
                p[apagar] = 0
                mudou = True
        if not mudou:
            return mascara[1:-1, 1:-1] * 255


def curvas_da_borda(bordas):
    """Percorre arestas do esqueleto, separando curvas nas bifurcações."""
    ys, xs = np.nonzero(bordas)
    pixels = set(zip(xs.tolist(), ys.tolist()))
    grafo = {}
    for x, y in sorted(pixels):
        adjacentes = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                q = (x + dx, y + dy)
                if (dx == dy == 0) or q not in pixels:
                    continue
                # Evita atalhos diagonais em cantos que já têm ligação ortogonal.
                if dx and dy and ((x + dx, y) in pixels or (x, y + dy) in pixels):
                    continue
                adjacentes.append(q)
        grafo[(x, y)] = adjacentes
    visitadas = set()
    inicios = sorted(grafo, key=lambda p: (len(grafo[p]) == 2, p))
    for inicio in inicios:
        for segundo in grafo[inicio]:
            aresta = tuple(sorted((inicio, segundo)))
            if aresta in visitadas:
                continue
            curva = [inicio]
            anterior, atual = inicio, segundo
            while True:
                visitadas.add(tuple(sorted((anterior, atual))))
                curva.append(atual)
                if atual == inicio or len(grafo[atual]) != 2:
                    break
                proximo = next(p for p in grafo[atual] if p != anterior)
                if tuple(sorted((atual, proximo))) in visitadas:
                    break
                anterior, atual = atual, proximo
            fechada = curva[-1] == curva[0]
            if fechada:
                curva.pop()
            if len(curva) >= 3:
                yield np.asarray(curva, dtype=np.float32), fechada


def _melhor_faixa(pontos):
    """Faixa dinâmica: cresce/rotaciona e retém a maior razão comprimento/largura.

    O retângulo orientado é ajustado geometricamente com minAreaRect. As
    larguras discretas são parâmetros numéricos, não parâmetros aprendidos.
    """
    if len(pontos) < 2:
        return 0., np.zeros(2), 0
    candidatos = []
    for fim in range(2, len(pontos) + 1):
        (_, _), (lado_a, lado_b), angulo = cv2.minAreaRect(pontos[:fim])
        comprimento, largura = max(lado_a, lado_b), min(lado_a, lado_b)
        if largura > LARGURAS_FAIXA[-1] + 1e-5:
            break
        rad = np.deg2rad(angulo + (90 if lado_b > lado_a else 0))
        direcao = np.array([np.cos(rad), np.sin(rad)])
        if np.dot(direcao, pontos[fim - 1] - pontos[0]) < 0:
            direcao *= -1
        candidatos.append((comprimento, largura, direcao, fim - 1))
    melhor = (0., np.zeros(2), 0)
    for largura_faixa in LARGURAS_FAIXA:
        ultimo = None
        for candidato in candidatos:
            if candidato[1] > largura_faixa + 1e-5:
                break
            ultimo = candidato
        if ultimo is not None and ultimo[0] / largura_faixa > melhor[0]:
            melhor = (ultimo[0] / largura_faixa, ultimo[2], ultimo[3])
    return melhor


def pontos_dominantes(curva, fechada=False, cancelar=None):
    """Duas faixas + mérito E_esq * |180 - ângulo| * E_dir (seção 2).

    Máximos locais, reposição de pontos não cobertos e eliminação de
    intermediários colineares seguem os três passos descritos no artigo.
    """
    n = len(curva)
    meritos = np.zeros(n)
    alcances = []
    for i in range(n):
        verificar_cancelamento(cancelar)
        if fechada:
            esquerda = (i - np.arange(n // 2 + 1)) % n
            direita = (i + np.arange(n // 2 + 1)) % n
        else:
            esquerda = np.arange(i, -1, -1)
            direita = np.arange(i, n)
        el, dl, il = _melhor_faixa(curva[esquerda])
        er, dr, ir = _melhor_faixa(curva[direita])
        if el and er:
            angulo = np.degrees(np.arccos(np.clip(np.dot(dl, dr), -1., 1.)))
            meritos[i] = el * abs(180. - angulo) * er
        alcances.append(set(esquerda[:il + 1]) | set(direita[:ir + 1]))
    selecionados, cobertos = set(), set()
    ordem = sorted(range(n), key=lambda i: (-meritos[i], i))
    for i in ordem:
        vizinhos = [(i - 1) % n, (i + 1) % n] if fechada else list(range(max(0, i - 1), min(n, i + 2)))
        if (meritos[i] > 0 and i not in cobertos
                and all(meritos[i] >= meritos[j] for j in vizinhos)):
            selecionados.add(i)
            cobertos.update(alcances[i])
    # Reposição onde as faixas dos máximos não cobriram a curva.
    for i in ordem:
        if i not in cobertos:
            selecionados.add(i)
            cobertos.update(alcances[i])
    if not fechada:
        selecionados.update((0, n - 1))
    mudou = True
    while mudou and len(selecionados) > (3 if fechada else 2):
        mudou = False
        indices = sorted(selecionados)
        for k in (range(len(indices)) if fechada else range(1, len(indices) - 1)):
            i = indices[k]
            a, p, b = curva[indices[k - 1]], curva[i], curva[indices[(k + 1) % len(indices)]]
            vetor = b - a
            norma = float(np.linalg.norm(vetor))
            if norma == 0:
                continue
            t = float(np.dot(p - a, vetor) / (norma * norma))
            desvio = abs(float(vetor[0] * (p - a)[1] - vetor[1] * (p - a)[0])) / norma
            if 0 <= t <= 1 and desvio <= LARGURAS_FAIXA[0]:
                selecionados.remove(i)
                mudou = True
                break
    ids = sorted(selecionados)
    return np.column_stack((curva[ids], meritos[ids]))


def bordas_do_rosto(rosto):
    """Etapa de contornos compartilhada pelo reconhecedor e pela visualização."""
    if rosto.shape[:2] != (TAMANHO[1], TAMANHO[0]):
        raise ValueError("Normalize a imagem pelos dois olhos antes da extração.")
    cinza = cv2.cvtColor(rosto, cv2.COLOR_BGR2GRAY) if rosto.ndim == 3 else rosto
    # A seção 2 permite qualquer detector de bordas; Canny dispensa modelos.
    return afinar_bordas(cv2.Canny(cinza, 80, 160))


def extrair_descritor(rosto, progresso=None, cancelar=None):
    """Recorte normalizado -> cinza -> bordas finas -> (x, y, mérito)."""
    bordas = bordas_do_rosto(rosto)
    curvas = list(curvas_da_borda(bordas))
    conjuntos = []
    for i, (curva, fechada) in enumerate(curvas):
        verificar_cancelamento(cancelar)
        conjuntos.append(pontos_dominantes(curva, fechada, cancelar))
        if progresso is not None:
            progresso((i + 1) / len(curvas), "Extraindo contornos do rosto...")
    if not conjuntos:
        raise ValueError("A captura não contém contornos suficientes. Ajuste a iluminação.")
    pontos = np.concatenate(conjuntos)
    # Bifurcações pertencem a mais de uma curva: não duplicar o mesmo pixel.
    unicos = {}
    for x, y, merito in pontos:
        unicos[(x, y)] = max(unicos.get((x, y), 0.), merito)
    pontos = np.array([(x, y, w) for (x, y), w in sorted(unicos.items())])
    if len(pontos) < 3 or pontos[:, 2].sum() <= 0:
        raise ValueError("A captura não contém pontos dominantes com mérito válido.")
    return pontos


def imagem_dos_contornos(rosto, descritor, escala=3):
    """Contornos cinza e círculos pretos nos pontos realmente usados no M²HD.

    Só renderiza: não extrai novos pontos, não muda o descritor e não faz
    ligação artificial entre pontos que pertencem a curvas diferentes.
    """
    if not isinstance(escala, int) or not 1 <= escala <= 8:
        raise ValueError("A escala da imagem deve ser um inteiro entre 1 e 8.")
    pontos = np.asarray(descritor, dtype=np.float64)
    if (pontos.ndim != 2 or pontos.shape[1] != 3 or not len(pontos)
            or not np.isfinite(pontos).all() or np.any(pontos[:, 2] < 0)
            or np.any(pontos[:, :2] < 0) or np.any(pontos[:, :2] >= TAMANHO)):
        raise ValueError("Use o descritor (x, y, mérito) do rosto normalizado.")
    bordas = bordas_do_rosto(rosto)
    # Coordenadas de pixels permanecem no mesmo sistema 160 x 200 do cálculo.
    desenho = np.full((TAMANHO[1] * escala, TAMANHO[0] * escala), 255, np.uint8)
    for curva, fechada in curvas_da_borda(bordas):
        vertices = np.rint((curva + .5) * escala).astype(np.int32)
        cv2.polylines(desenho, [vertices], fechada, 135, 1, cv2.LINE_AA)
    # Inclui pixels de curvas curtas/isoladas que não participam do traçado acima.
    ys, xs = np.nonzero(bordas)
    desenho[np.minimum((ys * escala + escala // 2), len(desenho) - 1),
            np.minimum((xs * escala + escala // 2), desenho.shape[1] - 1)] = 135
    for x, y, _ in pontos:
        centro = (int(round((x + .5) * escala)), int(round((y + .5) * escala)))
        cv2.circle(desenho, centro, max(1, escala), 0, 1, cv2.LINE_AA)
    return Image.fromarray(desenho)


def salvar_painel_contornos(caminho, amostras):
    """Exporta PNG para apresentação: lista de (título, recorte, descritor)."""
    if not amostras:
        raise ValueError("Faça uma comparação antes de exportar os contornos.")
    mapas = [(str(titulo), imagem_dos_contornos(rosto, pontos), len(pontos))
             for titulo, rosto, pontos in amostras]
    margem, espaco, cabecalho, rodape = 24, 24, 70, 50
    largura, altura = mapas[0][1].size
    painel = Image.new("RGB", (margem * 2 + len(mapas) * largura + (len(mapas) - 1) * espaco,
                               cabecalho + altura + rodape), "white")
    texto = ImageDraw.Draw(painel)
    def carregar_fonte(tamanho):
        for familia in ("arial.ttf", "DejaVuSans.ttf"):
            try:
                return ImageFont.truetype(familia, tamanho)
            except OSError:
                pass
        return ImageFont.load_default(size=tamanho)
    fonte = carregar_fonte(20)
    legenda = carregar_fonte(16)
    for i, (titulo, mapa, quantidade) in enumerate(mapas):
        x = margem + i * (largura + espaco)
        titulo_curto = titulo
        while texto.textlength(titulo_curto, font=fonte) > largura and len(titulo_curto) > 3:
            titulo_curto = titulo_curto[:-4] + "..."
        texto.text((x, 12), titulo_curto, fill="black", font=fonte)
        texto.text((x, 39), f"{quantidade} pontos dominantes", fill="#444444", font=legenda)
        painel.paste(mapa.convert("RGB"), (x, cabecalho))
    texto.text((margem, cabecalho + altura + 15),
               "Cinza: contornos afinados | Círculos pretos: pontos dominantes", fill="#333333", font=legenda)
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    painel.save(caminho, format="PNG", dpi=(300, 300))
    return str(caminho)


def distancia_m2hd(modelo, teste):
    """Equações (4) e (5): vizinho por distância; peso só depois da escolha.

    h(A,B) = sum(w_ij * ||a_i-b_j||) / sum(w_ij),
    j = argmin_j ||a_i-b_j||; w_ij = (mérito_i + mérito_j)/2.
    H(A,B) = max(h(A,B), h(B,A)). Unidade: pixels do recorte comum.
    """
    a, b = np.asarray(modelo, dtype=np.float64), np.asarray(teste, dtype=np.float64)
    for pontos in (a, b):
        if (pontos.ndim != 2 or pontos.shape[1] != 3 or len(pontos) == 0
                or not np.isfinite(pontos).all() or np.any(pontos[:, 2] < 0)):
            raise ValueError("Descritor inválido: use pontos (x, y, mérito não negativo).")
    def dirigida(origem, destino):
        numerador = denominador = 0.
        # Blocos limitam memória sem alterar os vizinhos ou a fórmula.
        for inicio in range(0, len(origem), 256):
            bloco = origem[inicio:inicio + 256]
            diferencas = bloco[:, None, :2] - destino[None, :, :2]
            quadrados = np.einsum("ijk,ijk->ij", diferencas, diferencas)
            vizinhos = np.argmin(quadrados, axis=1)
            distancias = np.sqrt(quadrados[np.arange(len(bloco)), vizinhos])
            pesos = (bloco[:, 2] + destino[vizinhos, 2]) / 2
            numerador += float(np.dot(pesos, distancias))
            denominador += float(pesos.sum())
        if denominador <= 0:
            raise ValueError("Não há méritos positivos para ponderar a comparação.")
        return numerador / denominador
    return max(dirigida(a, b), dirigida(b, a))


def classificar(descritor, referencias, nomes, progresso=None, cancelar=None):
    """Ranking fechado top-1 do artigo; não inventa um limiar de acesso."""
    if len(referencias) != len(nomes):
        raise ValueError("Cada referência precisa de um nome.")
    melhores = {}
    for i, (referencia, nome) in enumerate(zip(referencias, nomes)):
        verificar_cancelamento(cancelar)
        distancia = distancia_m2hd(referencia, descritor)
        melhores[nome] = min(melhores.get(nome, float("inf")), distancia)
        if progresso is not None:
            progresso((i + 1) / len(referencias), "Comparando com as fotos cadastradas...")
    return sorted(melhores.items(), key=lambda item: (item[1], item[0]))


def salvar_olhos(caminho_imagem, olhos):
    """Anota os olhos sem modificar a foto; um JSON por pessoa."""
    caminho = Path(caminho_imagem)
    arquivo = caminho.parent / ARQUIVO_GEOMETRIA
    dados = _ler_geometria(arquivo)
    dados[caminho.name] = np.asarray(olhos).tolist()
    _gravar_geometria(arquivo, dados)


def carregar_referencias(pasta, marcar_olhos=None, progresso=None, cancelar=None, origens=None):
    """Uma referência por foto; cache local validado pela foto, olhos e algoritmo."""
    referencias, nomes, avisos = [], [], []
    fotos = [foto for pessoa in sorted(Path(pasta).iterdir()) if pessoa.is_dir()
             for foto in sorted(pessoa.iterdir()) if foto.suffix.lower() in EXTENSOES] if Path(pasta).exists() else []
    for i, foto in enumerate(fotos):
        verificar_cancelamento(cancelar)
        pessoa = foto.parent
        arquivo = pessoa / ARQUIVO_GEOMETRIA
        dados = _ler_geometria(arquivo)
        try:
            olhos = dados.get(foto.name)
            if olhos is None and marcar_olhos is not None:
                olhos = marcar_olhos(ler_imagem(foto), f"Alinhar referência: {pessoa.name} / {foto.name}")
                verificar_cancelamento(cancelar)
                if olhos is not None:
                    salvar_olhos(foto, olhos)
            if olhos is None:
                avisos.append(f"{pessoa.name}/{foto.name}: faltam marcações dos olhos")
                continue
            stat = foto.stat()
            assinatura = [VERSAO_DESCRITOR, stat.st_mtime_ns, stat.st_size, np.asarray(olhos).tolist()]
            cache = dados.get("_descritores_m2hd", {})
            registro = cache.get(foto.name, {}) if isinstance(cache, dict) else {}
            descritor = None
            if isinstance(registro, dict) and registro.get("assinatura") == assinatura:
                try:
                    candidato = np.asarray(registro["pontos"], dtype=np.float64)
                    if (candidato.ndim == 2 and candidato.shape[1] == 3 and len(candidato) >= 3
                            and np.isfinite(candidato).all() and np.all(candidato[:, 2] >= 0)
                            and candidato[:, 2].sum() > 0):
                        descritor = candidato
                except (KeyError, TypeError, ValueError):
                    pass
            if descritor is None:
                def andamento(fracao, texto):
                    if progresso is not None:
                        progresso((i + fracao) / len(fotos), f"Preparando foto {i + 1}/{len(fotos)}: {pessoa.name}")
                descritor = extrair_descritor(normalizar_rosto(ler_imagem(foto), olhos), andamento, cancelar)
                verificar_cancelamento(cancelar)
                # Atualiza o JSON mais recente, preservando marcações feitas pela interface.
                atual = _ler_geometria(arquivo)
                if not isinstance(atual.get("_descritores_m2hd"), dict):
                    atual["_descritores_m2hd"] = {}
                atual["_descritores_m2hd"][foto.name] = {"assinatura": assinatura, "pontos": descritor.tolist()}
                try:
                    _gravar_geometria(arquivo, atual)
                except OSError:
                    pass  # Pasta somente leitura: ainda é possível comparar em memória.
            referencias.append(descritor)
            nomes.append(pessoa.name)
            if origens is not None:
                origens.append({"caminho": str(foto), "olhos": olhos})
        except (OSError, ValueError, TypeError, cv2.error) as erro:
            avisos.append(f"{pessoa.name}/{foto.name}: {erro}")
        finally:
            if progresso is not None:
                progresso((i + 1) / len(fotos), f"Referências prontas: {i + 1}/{len(fotos)}")
    return referencias, nomes, avisos
