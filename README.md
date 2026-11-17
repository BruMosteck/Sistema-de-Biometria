# Sistema de Biometria e Reconhecimento Facial

Trabalho de Álgebra Linear da ESEG — Grupo ETAPA: comparação de uma
representação geométrica definida por regras com representações aprendidas
por redes neurais.

| Versão | Reconhecimento | Entrada |
| --- | --- | --- |
| `semAPI` | Contornos, pontos dominantes e distância M²HD, sem modelos treinados. | `semAPI/app_biometria.py` |
| [`comAPI`](comAPI/README.md) | InsightFace e MediaPipe, com API local. | `comAPI/app_biometria.py` |

Ambas usam álgebra. A distinção é como representam o rosto, e não a presença
de uma API. A antiga implementação com `face_recognition` também usava ML;
ela foi substituída no `semAPI`.

## Executar a versão clássica

No PowerShell, a partir da raiz do repositório, com Python 3.12 e Tkinter:

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r .\semAPI\requirements.txt
& .\.venv\Scripts\python.exe .\semAPI\app_biometria.py
```

No VS Code, selecione o interpretador da `.venv`, abra
`semAPI/app_biometria.py` e use **Executar arquivo Python** (triângulo).
Não há servidor, download de pesos ou treinamento nessa versão.
Não é preciso desinstalar as dependências da versão com API para executar a clássica.

## Usar o aplicativo

1. **Inscrever Novo Usuário:** informe o nome e capture uma foto frontal,
   com o rosto inteiro e iluminação uniforme. Marque o centro dos dois olhos
   e confirme para concluir o cadastro. A guia da câmera é apenas visual, não um detector.
2. **Reconhecer por Contornos:** alinhe as referências antigas quando solicitado.
   Clique em **Comparar captura**, marque os olhos na imagem congelada e confirme.
   O aplicativo mostra o nome da melhor correspondência acima do vídeo, com
   a captura e a foto cadastrada lado a lado. **Ver detalhes da comparação**
   exibe o ranking e as distâncias em pixels, sem convertê-las em confiança.
   **Salvar contornos** exporta um PNG da captura e da referência lado a lado,
   com os contornos afinados em cinza e os pontos dominantes destacados em preto,
   como uma visualização das representações utilizadas na comparação.
   A pasta sugerida é `semAPI/exportacoes/`. A foto original não é modificada.
3. **Nova captura:** retorna à prévia ao vivo. A extração e a comparação da
   captura são executadas em uma thread, sem processar cada quadro da webcam.
   O carregamento da base também ocorre em segundo plano. Uma barra e um
   indicador de tempo mostram o andamento; falhas liberam uma nova tentativa.
4. **Armazenamento:** escolhe a pasta local da base. Cada pessoa tem uma pasta
   com fotos limpas e um `geometria_classica.json` com as marcações dos olhos
   e um cache dos pontos dominantes já calculados.
   As fotos existentes são preservadas; novas capturas não sobrescrevem as antigas.
5. **Análise de Simetria Facial:** usa seis pontos marcados manualmente. Essa
   ferramenta é independente do reconhecedor e não faz parte do método M²HD.

A marcação humana evita introduzir um detector treinado no fluxo. O aplicativo
nesta versão compara um rosto frontal por captura. Fotos antigas de perfil podem
ser ignoradas na marcação; o alinhamento de perfis por nariz/queixo não está
implementado. Não é realizada detecção automática de vários rostos.

O cache evita extrair novamente os contornos em cada abertura. Ele é invalidado
quando a foto, a marcação dos olhos ou a versão do algoritmo muda. Armazena
apenas resultados dos mesmos cálculos geométricos, sem treinamento ou modelos
de ML. A primeira preparação de cada foto ainda executa a extração completa.

## Algoritmo e vínculo com o artigo

Referência: **Gao, Y. (2003). Efficiently comparing face images using a modified
Hausdorff distance. IEE Proceedings – Vision, Image and Signal Processing,
150(6), 346–350. DOI: [10.1049/ip-vis:20030805](https://doi.org/10.1049/ip-vis:20030805).**
[Texto disponibilizado pelo autor](https://www.researchgate.net/publication/3359331_Efficiently_comparing_face_images_using_a_modified_Hausdorff_distance).

O reconhecedor segue os conceitos das seções 2–4:

1. Normalização e recorte pelos olhos, colocando todas as imagens no mesmo
   sistema de coordenadas.
2. Extração de bordas e afinamento para curvas de um pixel.
3. Pontos dominantes por duas faixas dinâmicas e mérito geométrico:
   `E_esquerda × |180° − ângulo| × E_direita`, onde `E` é comprimento/largura.
   Seleção por máximos, reposição de pontos descobertos e remoção de colineares.
4. Vizinho mais próximo por distância euclidiana; ponderação pela média dos
   méritos dos dois pontos. Distância dirigida conforme a equação (4):

   `h(A,B) = Σ(w_ij × ||a_i − b_j||) / Σ(w_ij)`

   com `j = argmin_j ||a_i − b_j||` e `w_ij = (mérito_i + mérito_j)/2`.
5. Comparação simétrica conforme a equação (5):

   `M²HD(A,B) = max(h(A,B), h(B,A))`.

Cada foto gera seu próprio conjunto de pontos, de tamanho variável. Não se faz
média de coordenadas entre fotos, pois não existe correspondência entre seus
pontos. O resultado de uma pessoa é a menor distância entre suas referências.
Nenhum descritor, peso de rede, covariância de classes ou classificador é treinado.

### Escolhas de implementação

- **Bordas:** Canny com limiares fixos 80/160, seguido de afinamento Zhang–Suen.
  O artigo permite qualquer detector de bordas; o experimento original usou
  Nevatia–Babu. O afinamento também é uma etapa prevista no método.
- **Duas faixas:** implementação própria do procedimento descrito para Dyn2S;
  os retângulos orientados são ajustados por `cv2.minAreaRect`. Larguras de
  0,5 a 2,5 pixels, em passos de 0,5. São escolhas numéricas fixas.
- **Alinhamento:** olhos marcados pelo usuário; transformação de similaridade
  por rotação, escala e translação. Recorte de 160 × 200 pixels, olhos em
  `(48, 52)` e `(112, 52)`. Recortes que saem da imagem são recusados.
- **Decisão:** ranking fechado top-1, como nos experimentos de identificação.
  Não há limiar universal para desconhecidos no artigo; por isso a interface
  mostra “mais próximo”, sem afirmar “acesso permitido” ou probabilidade.

Essas escolhas não introduzem outro método de reconhecimento ou ML. São
adaptações explícitas do processamento e da interação; não se afirma reproduzir
o código, os parâmetros completos ou as taxas de acerto do autor.

## Simetria manual da versão clássica

A análise de simetria pede seis marcações na foto frontal: os dois olhos,
a ponte do nariz, a ponta do queixo e duas extremidades equivalentes do
contorno do rosto. Esquerda e direita correspondem aos lados da fotografia.

A linha dos olhos define o alinhamento, e seu ponto médio define a origem.
O eixo central é perpendicular a essa linha. Após alinhar as coordenadas:

- O erro do contorno é a distância 2D entre uma extremidade refletida e a outra.
- Os erros do nariz e do queixo são as distâncias entre cada ponto e sua
  própria reflexão pelo eixo central.
- Cada erro é dividido pela largura entre as extremidades; a pontuação é
  `máximo(0, 100 × (1 − média dos três erros normalizados))`.

Os olhos servem de referência geométrica, sem uma nota própria. A altura
do queixo é exibida, mas mudar apenas sua altura não representa assimetria
bilateral. Rotação no plano, translação e escala uniforme não alteram a nota
quando as marcações correspondem aos mesmos pontos. A análise continua
didática, sensível à pose e às marcações, e separada do reconhecimento M²HD.

## Arquivos da versão clássica

| Arquivo | Responsabilidade |
| --- | --- |
| [`semAPI/app_biometria.py`](semAPI/app_biometria.py) | Interface, webcam, marcações, cadastro local, resultado e simetria manual. |
| [`semAPI/reconhecimento_classico.py`](semAPI/reconhecimento_classico.py) | Normalização, bordas, faixas, méritos, M²HD, ranking e leitura das referências. |
| [`semAPI/1_cadastro.py`](semAPI/1_cadastro.py) | Atalho para o cadastro do mesmo aplicativo. |
| [`semAPI/2_reconhecimento.py`](semAPI/2_reconhecimento.py) | Atalho para o reconhecimento do mesmo aplicativo. |
| [`semAPI/requirements.txt`](semAPI/requirements.txt) | NumPy, OpenCV e Pillow; Tkinter acompanha o Python. |
| [`semAPI/test_reconhecimento_classico.py`](semAPI/test_reconhecimento_classico.py) | Testes matemáticos, imagens sintéticas, arquivos locais e ausência de imports de ML. |
| [`semAPI/Biometria ESEG.spec`](semAPI/Biometria%20ESEG.spec) | Empacotamento sem modelos de reconhecimento. |

## Verificação

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s .\semAPI -v
```

Os testes usam imagens sintéticas e não medem a precisão biométrica em pessoas.
Uma avaliação de reconhecimento exige imagens de teste independentes das
referências, com as mesmas condições de captura nas duas versões.

## Versões históricas

As capturas em `docs/imagens/` e o pacote Windows `v1.0.0` nas
[Releases](https://github.com/BruMosteck/Sistema-de-Biometria/releases)
correspondem à implementação anterior com modelos treinados. Para testar o
método clássico descrito acima, execute o código-fonte atualizado.

## Participantes

| Participante | GitHub |
| --- | --- |
| Bruno Simcka Mosteck | [@BruMosteck](https://github.com/BruMosteck) |
| Miguel Narciso Freitas | [@DevFreitasMiguel](https://github.com/DevFreitasMiguel) |
| Henrique Pinheiro | [@PinheiroLim](https://github.com/PinheiroLim) |
