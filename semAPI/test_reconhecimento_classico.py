r"""Verificações matemáticas e de integração, sem câmera ou dados pessoais.

Executar: .venv\Scripts\python.exe -m unittest discover -s semAPI -v
"""
import ast
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from concurrent.futures import CancelledError
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

import reconhecimento_classico as r


def rosto_sintetico(variante=0):
    imagem = np.zeros((200, 160, 3), np.uint8)
    cv2.ellipse(imagem, (80, 100), (55, 85), 0, 0, 360, (230, 230, 230), 2)
    for x in (48, 112):
        cv2.ellipse(imagem, (x, 52), (12, 5 + variante * 3), 0, 0, 360, (255, 255, 255), 2)
    cv2.line(imagem, (80, 70), (70 + variante * 18, 110), (255, 255, 255), 2)
    cv2.ellipse(imagem, (80, 140), (22 + variante * 8, 8), 0, 0, 180, (255, 255, 255), 2)
    return imagem


class DistanciaTest(unittest.TestCase):
    def test_equacoes_4_e_5_com_resultado_calculado_a_mao(self):
        a = np.array([[0, 0, 1], [4, 0, 3]])
        b = np.array([[1, 0, 5], [8, 0, 7]])
        # h(A,B)=15/7; h(B,A)=23/8. A distância não é a média desses dois.
        self.assertAlmostEqual(r.distancia_m2hd(a, b), 23 / 8)
        self.assertAlmostEqual(r.distancia_m2hd(b, a), 23 / 8)

    def test_identidade_translacao_e_escala(self):
        a = np.array([[0., 0, 1], [20, 0, 2], [0, 20, 3]])
        b = a.copy()
        b[:, :2] += [3, 4]
        self.assertEqual(r.distancia_m2hd(a, a), 0)
        self.assertAlmostEqual(r.distancia_m2hd(a, b), 5)
        a[:, :2] *= 2
        b[:, :2] *= 2
        self.assertAlmostEqual(r.distancia_m2hd(a, b), 10)

    def test_merito_nao_escolhe_vizinho(self):
        # O ponto mais próximo deve ser escolhido antes de se aplicar seu peso.
        a = np.array([[0., 0, 1]])
        b = np.array([[1., 0, 100], [2., 0, 1]])
        self.assertAlmostEqual(r.distancia_m2hd(a, b), 52.5 / 51.5)

    def test_descritores_invalidos(self):
        valido = np.array([[0., 0, 1]])
        for invalido in ([], [[1, 2]], [[0, 0, -1]], [[0, np.nan, 1]], [[0, 0, 0]]):
            with self.subTest(invalido=invalido), self.assertRaises(ValueError):
                r.distancia_m2hd(invalido, invalido)
        with self.assertRaises(ValueError):
            r.classificar(valido, [valido], [])

    def test_ranking_compara_fotos_sem_media_de_coordenadas(self):
        consulta = np.array([[0., 0, 1]])
        modelos = [np.array([[50., 0, 1]]), consulta.copy(), np.array([[3., 0, 1]])]
        ranking = r.classificar(consulta, modelos, ["Ana", "Ana", "Bia"])
        self.assertEqual(ranking, [("Ana", 0.), ("Bia", 3.)])
        self.assertEqual(r.classificar(consulta, [], []), [])


class GeometriaTest(unittest.TestCase):
    def test_olhos_transformam_exatamente_para_os_pontos_de_referencia(self):
        imagem = np.zeros((480, 640, 3), np.uint8)
        for olhos in ([[288, 152], [352, 152]], [[350, 165], [280, 145]]):
            matriz = r.matriz_alinhamento(imagem, olhos)
            pontos = np.array(sorted(olhos))
            destino = np.column_stack((pontos, np.ones(2))) @ matriz.T
            np.testing.assert_allclose(destino, r.OLHOS_DESTINO, atol=1e-10)
            self.assertEqual(r.normalizar_rosto(imagem, olhos).shape, (200, 160, 3))

    def test_alinhamento_rejeita_coincidencias_e_preenchimento_artificial(self):
        imagem = np.zeros((480, 640, 3), np.uint8)
        for olhos in ([[100, 100], [100, 100]], [[0, 0], [30, 0]], [[-1, 80], [60, 80]]):
            with self.assertRaises(ValueError):
                r.normalizar_rosto(imagem, olhos)

    def test_curvas_cobrem_as_arestas_sem_perder_ciclo_ou_ramo(self):
        mapa = np.zeros((24, 24), np.uint8)
        cv2.rectangle(mapa, (2, 2), (8, 8), 255, 1)
        cv2.line(mapa, (14, 4), (14, 18), 255)
        cv2.line(mapa, (14, 10), (20, 10), 255)
        curvas = list(r.curvas_da_borda(mapa))
        self.assertTrue(any(fechada for _, fechada in curvas))
        cobertos = {tuple(p) for curva, _ in curvas for p in curva}
        self.assertEqual(cobertos, {tuple(p[::-1]) for p in np.argwhere(mapa)})

    def test_duas_faixas_destacam_canto_e_preservam_extremos(self):
        curva = np.array([[x, 0] for x in range(11)] + [[10, y] for y in range(1, 11)], np.float32)
        descritor = r.pontos_dominantes(curva)
        self.assertIn((0., 0.), [tuple(p) for p in descritor[:, :2]])
        self.assertIn((10., 10.), [tuple(p) for p in descritor[:, :2]])
        canto = descritor[np.argmax(descritor[:, 2]), :2]
        np.testing.assert_array_equal(canto, [10, 0])
        self.assertLess(len(descritor), len(curva) / 2)

    def test_afinamento_preserva_componente_e_e_idempotente(self):
        mapa = np.zeros((30, 30), np.uint8)
        cv2.line(mapa, (5, 15), (25, 15), 255, 5)
        fino = r.afinar_bordas(mapa)
        self.assertLess(np.count_nonzero(fino), np.count_nonzero(mapa))
        self.assertGreater(np.count_nonzero(fino), 0)
        self.assertEqual(cv2.connectedComponents(fino)[0], 2)
        np.testing.assert_array_equal(r.afinar_bordas(fino), fino)


class SimetriaManualTest(unittest.TestCase):
    def setUp(self):
        from app_biometria import calcular_simetria_manual
        self.calcular = calcular_simetria_manual
        self.pontos = np.array([[-20., 0], [20, 0], [0, 15], [0, 80], [-50, 40], [50, 40]])

    def test_reflexao_perfeita_e_altura_central_nao_sao_penalizadas(self):
        resultado = self.calcular(self.pontos)
        self.assertAlmostEqual(resultado["pontuacao"], 100)
        self.assertAlmostEqual(resultado["largura"], 100)
        self.pontos[3, 1] = 120
        self.assertAlmostEqual(self.calcular(self.pontos)["pontuacao"], 100)

    def test_desvios_do_nariz_queixo_e_desnivel_do_contorno_entram_na_nota(self):
        self.pontos[2, 0] = 9
        self.pontos[3, 0] = 15
        self.pontos[5, 1] = 60
        resultado = self.calcular(self.pontos)
        self.assertAlmostEqual(resultado["pontuacao"], 100 * (1 - (.18 + .30 + .20) / 3))
        self.assertAlmostEqual(resultado["erros_normalizados"]["contorno"], .20)
        self.assertEqual(resultado["queixo_alinhado"], (15., 80.))

    def test_rotacao_escala_e_translacao_preservam_a_pontuacao(self):
        self.pontos[3, 0] = 15
        esperado = self.calcular(self.pontos)["pontuacao"]
        angulo = np.radians(30)
        rotacao = np.array([[np.cos(angulo), -np.sin(angulo)], [np.sin(angulo), np.cos(angulo)]])
        alterados = (self.pontos @ rotacao.T) * 3 + [300, 120]
        resultado = self.calcular(alterados)
        self.assertAlmostEqual(resultado["pontuacao"], esperado)
        self.assertAlmostEqual(resultado["largura"], 300)
        self.assertAlmostEqual(resultado["desnivel_extremidades"], 0)

    def test_marcacoes_incompletas_invertidas_e_degeneradas_sao_rejeitadas(self):
        invalidos = [self.pontos[:5], np.full((6, 2), np.nan)]
        for indice, valor in ((1, [-20, 0]), (3, [0, -10]), (3, [90, 80]),
                              (4, [60, 40]), (5, [-40, 40])):
            pontos = self.pontos.copy()
            pontos[indice] = valor
            invalidos.append(pontos)
        for pontos in invalidos:
            with self.subTest(pontos=pontos), self.assertRaises(ValueError):
                self.calcular(pontos)

    def test_resultado_da_aplicacao_expoe_queixo_e_extremidades(self):
        from app_biometria import BiometriaApp
        app = BiometriaApp.__new__(BiometriaApp)
        app.pontos_simetria = self.pontos.tolist()
        app.thread_calcular_simetria_matematica(None)
        self.assertFalse(app.resultado_matematico["erro"])
        self.assertIn("queixo", app.resultado_matematico["dados"][0])
        self.assertIn("extremidades", app.resultado_matematico["dados"][0])
        app.pontos_simetria = self.pontos[:5].tolist()
        app.thread_calcular_simetria_matematica(None)
        self.assertTrue(app.resultado_matematico["erro"])


class IntegracaoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.imagem = rosto_sintetico()
        cls.descritor = r.extrair_descritor(cls.imagem)

    def test_extracao_deterministica_compacta_e_sem_modelos(self):
        with patch.dict(sys.modules, {nome: None for nome in
                        ("dlib", "face_recognition", "insightface", "mediapipe", "torch", "onnxruntime")}):
            np.testing.assert_array_equal(r.extrair_descritor(self.imagem), self.descritor)
        bordas = r.afinar_bordas(cv2.Canny(cv2.cvtColor(self.imagem, cv2.COLOR_BGR2GRAY), 80, 160))
        self.assertGreater(self.descritor[:, 2].sum(), 0)
        self.assertLess(len(self.descritor), np.count_nonzero(bordas))
        for x, y, _ in self.descritor:
            self.assertNotEqual(bordas[int(y), int(x)], 0)

    def test_outra_forma_tem_distancia_maior_que_a_propria_referencia(self):
        outra = r.extrair_descritor(rosto_sintetico(1))
        ranking = r.classificar(self.descritor, [outra, self.descritor], ["Bia", "Ana"])
        self.assertEqual(ranking[0], ("Ana", 0.))
        self.assertGreater(ranking[1][1], 0)

    def test_foto_sem_contornos_nao_vira_referencia(self):
        with self.assertRaises(ValueError):
            r.extrair_descritor(np.zeros((200, 160, 3), np.uint8))

    def test_exportar_png_reutiliza_pontos_sem_modificar_o_reconhecedor(self):
        imagem_original, pontos_originais = self.imagem.copy(), self.descritor.copy()
        with tempfile.TemporaryDirectory() as temporario:
            caminho = Path(temporario) / "exportacoes" / "comparação.png"
            with patch.object(r, "extrair_descritor", side_effect=AssertionError("Não reextrair pontos")):
                r.salvar_painel_contornos(caminho, [
                    ("Captura analisada", self.imagem, self.descritor),
                    ("Referência", self.imagem, self.descritor)])
            with Image.open(caminho) as png:
                self.assertEqual(png.format, "PNG")
                self.assertEqual(png.size, (1032, 720))
                self.assertAlmostEqual(png.info["dpi"][0], 300, delta=.1)
                self.assertLess(np.array(png).min(), 255)
        np.testing.assert_array_equal(self.imagem, imagem_original)
        np.testing.assert_array_equal(self.descritor, pontos_originais)

    def test_exportacao_rejeita_pontos_fora_do_recorte(self):
        with self.assertRaises(ValueError):
            r.imagem_dos_contornos(self.imagem, [[200, 250, 1]])
        with self.assertRaises(ValueError):
            r.salvar_painel_contornos("nao_criar.png", [])

    def test_referencias_antigas_marcacao_persistente_e_arquivo_corrompido(self):
        with tempfile.TemporaryDirectory() as temporario:
            pessoa = Path(temporario) / "João"
            pessoa.mkdir()
            imagem = np.zeros((480, 640, 3), np.uint8)
            imagem[100:300, 240:400] = self.imagem
            foto = pessoa / "foto_1.PNG"
            Image.fromarray(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)).save(foto)
            conteudo_original = foto.read_bytes()
            (pessoa / "foto_ruim.jpg").write_bytes(b"arquivo invalido")
            with patch.object(r, "salvar_olhos", wraps=r.salvar_olhos) as salvar:
                referencias, nomes, avisos = r.carregar_referencias(temporario,
                    lambda imagem, titulo: [[288, 152], [352, 152]])
                self.assertEqual(salvar.call_count, 1)
            self.assertEqual(nomes, ["João"])
            self.assertEqual(len(avisos), 1)
            np.testing.assert_array_equal(referencias[0], self.descritor)
            with patch.object(r, "salvar_olhos", side_effect=AssertionError("Não remarcar")):
                self.assertEqual(r.carregar_referencias(temporario)[1], ["João"])
            self.assertEqual(foto.read_bytes(), conteudo_original)

    def test_cancelar_marcacao_nao_cria_referencia_nem_modifica_foto(self):
        with tempfile.TemporaryDirectory() as temporario:
            pessoa = Path(temporario) / "Ana"
            pessoa.mkdir()
            foto = pessoa / "foto.png"
            Image.fromarray(self.imagem).save(foto)
            referencias, nomes, avisos = r.carregar_referencias(temporario, lambda *args: None)
            self.assertEqual((referencias, nomes), ([], []))
            self.assertEqual(len(avisos), 1)
            self.assertFalse((pessoa / r.ARQUIVO_GEOMETRIA).exists())

    def test_cache_evitar_reextracao_e_invalidar_ao_alterar_olhos_ou_foto(self):
        with tempfile.TemporaryDirectory() as temporario:
            pessoa = Path(temporario) / "Ana"
            pessoa.mkdir()
            foto = pessoa / "foto.png"
            imagem = np.zeros((480, 640, 3), np.uint8)
            imagem[100:300, 240:400] = self.imagem
            Image.fromarray(imagem).save(foto)
            olhos = [[288, 152], [352, 152]]
            r.salvar_olhos(foto, olhos)
            # Fixture da extração isola a invalidação do cache dos cálculos já testados.
            with patch.object(r, "extrair_descritor", return_value=self.descritor) as extrair:
                primeira = r.carregar_referencias(temporario)
                segunda = r.carregar_referencias(temporario)
                self.assertEqual(extrair.call_count, 1)
                np.testing.assert_array_equal(primeira[0][0], segunda[0][0])
                r.salvar_olhos(foto, [[289, 152], [353, 152]])
                r.carregar_referencias(temporario)
                self.assertEqual(extrair.call_count, 2)
                Image.fromarray(imagem + 1).save(foto)
                r.carregar_referencias(temporario)
                self.assertEqual(extrair.call_count, 3)
            with patch.object(r, "ler_imagem", side_effect=AssertionError("Cache não precisa reler a foto")):
                self.assertEqual(r.carregar_referencias(temporario)[1], ["Ana"])

    def test_cancelar_extracao_interrompe_sem_alterar_a_formula(self):
        cancelar = threading.Event()
        cancelar.set()
        with self.assertRaises(CancelledError):
            r.extrair_descritor(self.imagem, cancelar=cancelar)
        with self.assertRaises(CancelledError):
            r.classificar(self.descritor, [self.descritor], ["Ana"], cancelar=cancelar)

    def test_aplicativo_importa_com_bibliotecas_de_ml_bloqueadas(self):
        pasta = Path(__file__).resolve().parent
        codigo = (
            "import sys; "
            "sys.modules.update({n: None for n in "
            "('dlib','face_recognition','face_recognition_models','insightface','mediapipe','torch','onnxruntime')}); "
            "import app_biometria; print('OK')"
        )
        resultado = subprocess.run([sys.executable, "-c", codigo], cwd=pasta,
                                   capture_output=True, text=True, timeout=30)
        self.assertEqual(resultado.returncode, 0, resultado.stderr)
        self.assertIn("OK", resultado.stdout)
        proibidas = {"dlib", "face_recognition", "face_recognition_models", "insightface", "mediapipe"}
        for arquivo in pasta.glob("*.py"):
            arvore = ast.parse(arquivo.read_text(encoding="utf-8"))
            for no in ast.walk(arvore):
                if isinstance(no, ast.Import):
                    self.assertFalse(proibidas.intersection(a.name.split(".")[0] for a in no.names))
                elif isinstance(no, ast.ImportFrom):
                    self.assertNotIn((no.module or "").split(".")[0], proibidas)


class InterfaceTest(unittest.TestCase):
    """Tkinter real com janela oculta; webcam e cliques substituídos por fixtures."""
    def setUp(self):
        from app_biometria import BiometriaApp
        self.temporario = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporario.cleanup)
        self.root = tk.Tk()
        self.root.withdraw()
        with patch.object(BiometriaApp, "carregar_configuracao", return_value=self.temporario.name):
            self.app = BiometriaApp(self.root)
        self.addCleanup(self.app.fechar)
        self.frame = np.zeros((480, 640, 3), np.uint8)
        self.frame[100:300, 240:400] = rosto_sintetico()
        self.olhos = [[288, 152], [352, 152]]

    def camera_falsa(self):
        self.app.is_camera_running = True
        self.app.frame_atual_camera = self.frame.copy()

    def aguardar(self):
        limite = time.monotonic() + 20
        while self.app.comparando and time.monotonic() < limite:
            self.root.update()
            time.sleep(.01)
        self.assertFalse(self.app.comparando, "O processamento ficou sem resultado ou erro visível")

    def abrir_reconhecimento(self, descritor):
        with patch("app_biometria.carregar_referencias", return_value=([descritor], ["Ana"], [])), \
                patch.object(self.app, "iniciar_camera_assincrona", side_effect=self.camera_falsa):
            self.app.iniciar_verificacao()
            self.aguardar()

    def test_comparacao_assincrona_resultado_nova_captura_e_cancelamento(self):
        descritor = r.extrair_descritor(rosto_sintetico())
        self.abrir_reconhecimento(descritor)
        with patch.object(self.app, "marcar_olhos", return_value=None):
            self.app.comparar_captura()
        self.assertIsNone(self.app.captura_verificacao)
        with patch.object(self.app, "marcar_olhos", return_value=self.olhos):
            self.app.comparar_captura()
        # Mesmo se a webcam desconectar depois, a captura já feita é comparável.
        self.app.frame_atual_camera = None
        self.aguardar()
        self.assertIn("Melhor correspondência: Ana", self.app.lbl_comparacao.cget("text"))
        self.assertEqual(self.app.resultado_comparacao[0][0], ("Ana", 0.))
        self.assertIsNotNone(self.app.imagem_resultado)
        self.assertEqual(self.app.btn_detalhes.cget("state"), "normal")
        self.assertEqual(self.app.btn_salvar_contornos.cget("state"), "normal")
        self.assertEqual(len(self.app.mapas_comparacao), 1)
        # O cenário simulado não tem caminho da referência: exporta só a captura real.
        with patch("app_biometria.BASE_DIR", self.temporario.name), \
                patch("app_biometria.filedialog.asksaveasfilename", return_value=""):
            self.app.salvar_contornos_comparacao()
        self.app.nova_captura()
        self.assertIsNone(self.app.captura_verificacao)
        self.assertIsNone(self.app.resultado_comparacao)
        self.assertIsNone(self.app.mapas_comparacao)
        self.assertEqual(self.app.btn_salvar_contornos.cget("state"), "disabled")
        self.app.construir_menu()
        self.assertFalse(self.app.is_camera_running)

    def test_resultado_de_sessao_encerrada_nao_contamina_tela_nova(self):
        from app_biometria import extrair_descritor
        liberado, entrou = threading.Event(), threading.Event()
        def extrair_devagar(imagem):
            entrou.set()
            liberado.wait(timeout=10)
            return extrair_descritor(imagem)
        self.addCleanup(liberado.set)
        self.abrir_reconhecimento(np.array([[0., 0, 1]]))
        with patch.object(self.app, "marcar_olhos", return_value=self.olhos), \
                patch("app_biometria.extrair_descritor", side_effect=lambda imagem, **kwargs: extrair_devagar(imagem)):
            self.app.comparar_captura()
            self.assertTrue(entrou.wait(timeout=5))
            # Aguarda a conclusão real da thread para verificar a guarda de sessão.
            worker = self.app.thread_processamento
            self.app.construir_menu()
            liberado.set()
            worker.join(timeout=10)
        self.assertFalse(worker.is_alive())
        self.assertIsNone(self.app.resultado_comparacao)
        self.assertFalse(self.app.comparando)

    def test_carregamento_lento_mantem_tkinter_responsivo(self):
        liberado = threading.Event()
        self.addCleanup(liberado.set)
        def carregar(*args, **kwargs):
            liberado.wait(timeout=5)
            return [], [], []
        with patch("app_biometria.carregar_referencias", side_effect=carregar), \
                patch.object(self.app, "iniciar_camera_assincrona", side_effect=self.camera_falsa):
            self.app.iniciar_verificacao()
            self.assertTrue(self.app.comparando)
            recebeu_evento = []
            self.root.after(1, lambda: recebeu_evento.append(True))
            limite = time.monotonic() + 1
            while not recebeu_evento and time.monotonic() < limite:
                self.root.update()
            self.assertTrue(recebeu_evento, "A UI ficou bloqueada durante a leitura da base")
            liberado.set()
            self.aguardar()
        self.assertIn("Nenhuma foto válida", self.app.lbl_comparacao.cget("text"))

    def test_erro_inesperado_na_thread_aparece_e_libera_nova_tentativa(self):
        self.abrir_reconhecimento(np.array([[0., 0, 1]]))
        with patch.object(self.app, "marcar_olhos", return_value=self.olhos), \
                patch("app_biometria.extrair_descritor", side_effect=RuntimeError("Falha de teste")), \
                patch("traceback.print_exc"):
            self.app.comparar_captura()
            self.aguardar()
        self.assertIn("Não foi possível concluir", self.app.lbl_comparacao.cget("text"))
        self.assertIn("Falha de teste", self.app.lbl_andamento.cget("text"))
        self.assertEqual(self.app.btn_nova_captura.cget("state"), "normal")
        self.app.nova_captura()
        self.assertIsNone(self.app.captura_verificacao)

    def test_marcacao_de_referencia_e_executada_na_thread_do_tkinter(self):
        thread_marcacao = []
        def marcar(*args):
            thread_marcacao.append(threading.current_thread())
            return self.olhos
        def carregar(*args, **kwargs):
            self.assertNotEqual(threading.current_thread(), threading.main_thread())
            self.assertEqual(kwargs["marcar_olhos"](self.frame, "Marcar referência"), self.olhos)
            return [], [], []
        with patch("app_biometria.carregar_referencias", side_effect=carregar), \
                patch.object(self.app, "iniciar_camera_assincrona", side_effect=self.camera_falsa), \
                patch.object(self.app, "marcar_olhos", side_effect=marcar):
            self.app.iniciar_verificacao()
            self.aguardar()
        self.assertEqual(thread_marcacao, [threading.main_thread()])

    def test_resultado_e_controles_cabem_na_janela_padrao(self):
        self.abrir_reconhecimento(np.array([[0., 0, 1]]))
        self.app.lbl_comparacao.config(text="Melhor correspondência: Ana")
        self.app.lbl_andamento.config(text="Comparação concluída. Veja a captura e a foto cadastrada abaixo.")
        self.root.update_idletasks()
        self.assertLessEqual(self.root.winfo_reqheight(), 750)
        widgets = self.root.pack_slaves()
        self.assertLess(widgets.index(self.app.lbl_comparacao), widgets.index(self.app.lbl_video))

    def test_simetria_com_seis_pontos_mostra_novas_medidas_sem_cortar_controles(self):
        from PIL import ImageTk
        self.app.limpar_janela()
        self.app.canvas = tk.Canvas(self.root)
        self.app.canvas.pack()
        self.app.pontos_simetria = [[260, 150], [340, 150], [300, 190],
                                   [300, 330], [210, 240], [390, 240]]
        self.app.fotos_animacao = [ImageTk.PhotoImage(Image.new("RGB", (200, 200)))]
        self.app.thread_calcular_simetria_matematica(None)
        self.app.mostrar_resultados_simetria()
        self.root.update_idletasks()
        textos = []
        def visitar(widget):
            if isinstance(widget, tk.Label):
                textos.append(widget.cget("text"))
            for filho in widget.winfo_children():
                visitar(filho)
        visitar(self.root)
        self.assertTrue(any(t.startswith("Queixo:") for t in textos))
        self.assertTrue(any(t.startswith("Extremidades:") for t in textos))
        self.assertFalse(any("Maxilar" in t for t in textos))
        self.assertLessEqual(self.root.winfo_reqheight(), 750)
        self.assertLessEqual(self.root.winfo_reqwidth(), 900)


if __name__ == "__main__":
    unittest.main()
