# Biometria com InsightFace + MediaPipe

**Abra `app_biometria.py` e use ▶ no VS Code.** Ele abre a interface e inicia
a API local automaticamente, na mesma execução. A janela principal só aparece
quando os modelos e a API estão prontos, com os botões habilitados. Se houver
falha na inicialização, um aviso permite tentar novamente ou encerrar.
Não é necessário outro terminal
para iniciar o servidor. Ao fechar a janela, o servidor é encerrado.

São apenas dois arquivos Python: `app_biometria.py` (interface e cliente HTTP)
e `api_biometria.py` (API, reconhecimento, simetria e armazenamento).

## Instalação única

Python 3.12 de 64 bits com Tkinter. Na raiz do projeto:

```powershell
py -3.12 -m venv comAPI/.venv
& ./comAPI/.venv/Scripts/python.exe -m pip install -r ./comAPI/requirements.txt
```

No VS Code, selecione **Python: Select Interpreter** →
`comAPI/.venv/Scripts/python.exe`. Depois basta executar `comAPI/app_biometria.py`
pelo ▶. Alternativamente:

```powershell
& ./comAPI/.venv/Scripts/python.exe ./comAPI/app_biometria.py
```

O primeiro uso baixa o pacote `buffalo_l` do InsightFace e o modelo Face Landmarker
do MediaPipe. Acompanhe o terminal; pode levar vários minutos. Os modelos ficam
em `comAPI/modelos/` e são reutilizados. A inferência roda em CPU, localmente.
Depois de baixar os modelos, não é preciso enviar imagens para serviços externos.

## Funções

- Cadastro guiado: nome e três fotos (frente e leves rotações para cada lado).
  As instruções de ângulo são orientativas; o cadastro valida um único rosto e tamanho mínimo.
  O conjunto é salvo somente depois que as três fotos passam na validação.
- Reconhecimento ao vivo com InsightFace e malha de polígonos do MediaPipe.
  A foto salva não contém os desenhos. Quando a malha não é detectada,
  isso não impede o InsightFace de tentar reconhecer o rosto.
- Comparação com cada amostra individual: cosseno mínimo de 0,50 e diferença
  mínima de 0,05 para a segunda identidade. São parâmetros iniciais para avaliação,
  não limiares calibrados para produção nem porcentagens de certeza.
- Listagem e exclusão dos usuários e suas fotos.
- Seleção da pasta local dos cadastros. Trocar a pasta não move nem apaga a base anterior.
  É possível selecionar o `dataset` da versão antiga; vetores são recalculados com InsightFace.
- Simetria de foto cadastrada ou avulsa, sem cadastrar a foto avulsa.
  Exige um rosto frontal, sem obstruções e com expressão neutra; a pose é verificada
  pelo modelo. Não há simulação por espelhamento nesta etapa.

Fotos ficam em `dataset/<nome>/foto_1.jpg`, `foto_2.jpg`, `foto_3.jpg`, dentro de
`comAPI/` por padrão. A pasta escolhida é registrada em `config_biometria.json`.
Os vetores ficam em memória e são extraídos novamente ao abrir a aplicação.
Fotos inválidas ou com múltiplos rostos da base existente são ignoradas com aviso.
`dataset/`, configuração, ambientes virtuais e modelos não devem entrar no Git.

## Simetria: como é calculada

Os pontos 2D são alinhados pela linha dos olhos. O eixo central usa testa e queixo.
Comparamos pares de pontos dos olhos, sobrancelhas, boca e mandíbula com suas
posições espelhadas. A distância de cada par é dividida pela largura facial.
A pontuação é `máximo(0, 100 × (1 − média dos erros normalizados))`.

É uma métrica geométrica didática da fotografia, não uma avaliação clínica ou uma
medida universal de beleza. Pose, expressão, perspectiva, iluminação e erros de
localização dos pontos alteram o resultado. As pontuações diferem da fórmula da versão antiga.

## API local

O aplicativo escolhe uma porta livre em `127.0.0.1` e imprime o endereço no terminal.
As chamadas usam um token aleatório por execução, no cabeçalho `X-Biometria-Token`,
compartilhado internamente entre a interface e a API. Não há servidor público.

| Método | Rota | Função |
|---|---|---|
| GET | `/estado` | Estado, pasta e usuários |
| POST | `/preparar` | Carregar os modelos e extrair referências |
| POST | `/analisar` | Analisar uma imagem no campo multipart `arquivo` |
| POST | `/usuarios/{nome}/cadastro` | Cadastrar três arquivos multipart `fotos` |
| DELETE | `/usuarios/{nome}` | Excluir cadastro e fotos |
| GET | `/usuarios/{nome}/foto` | Obter a primeira foto para simetria |
| PUT | `/armazenamento` | Selecionar a pasta por JSON `pasta_armazenamento` |
| POST | `/simetria` | Analisar foto multipart `arquivo` |

Uploads aceitam JPEG, PNG, WEBP ou BMP de até 8 MiB e 20 milhões de pixels;
orientação EXIF é corrigida e o maior lado é reduzido para até 960 pixels.
Análises são serializadas para evitar uso simultâneo dos modelos. A prévia da webcam
é independente da API, com atualização programada para até 30 FPS (conforme a câmera
e o computador). O reconhecimento roda em segundo plano, com uma solicitação por vez,
sem acumular quadros. O fluxo óptico acompanha os polígonos entre as leituras reais;
não calcula identidade nem alimenta o cadastro. Novas malhas são alinhadas ao quadro
atual e suavizadas antes da troca. Uma falha isolada de detecção não apaga o desenho
ou o nome: duas leituras consecutivas sem rosto limpam a identificação. Uma nova
leitura com rosto desconhecido substitui imediatamente o nome anterior.
As sobreposições têm validade visual de 1,5 a 3 segundos desde o recebimento da
última leitura válida, ajustada ao intervalo das análises. Se a comunicação falhar,
são limpas. As fotos de cadastro continuam exigindo análise recente e usam o quadro
limpo da análise correspondente, independentemente dessa tolerância visual.

As sessões ONNX usam até quatro threads para deixar recursos para a interface e o
MediaPipe. A prévia do cadastro só detecta o rosto; os vetores das três fotos continuam
sendo extraídos e validados ao salvar. O modelo, a resolução do detector e os limiares
de reconhecimento permanecem iguais.

## Limites e modelos

Rostos cobertos, perfil extremo, baixa luz e fotos pequenas podem impedir o reconhecimento.
Não reduzimos automaticamente o limiar para forçar uma identificação. A malha é uma
estimativa e pode inferir pontos ocultos; não é uma medição de profundidade real.
O reconhecimento desta versão não inclui prova de presença real nem proteção validada contra fotos.

- [InsightFace](https://github.com/deepinsight/insightface/tree/master/python-package):
  código MIT; os pesos pré-treinados fornecidos são para pesquisa não comercial.
- [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker):
  pontos e geometria facial. A API de tarefas é usada, sem depender de `mp.solutions`.

As funções equivalentes à versão `semAPI/` foram preservadas; o motor e o cálculo
de simetria são novos. A precisão com máscara e perfil precisa ser avaliada com
amostras de teste separadas do cadastro, incluindo pessoas desconhecidas.
