import sys
import os
import types
import importlib.util
import numpy as np

# Monkey Patch para corrigir o erro 'pkg_resources' no Python 3.14+
mock_pkg = types.ModuleType("pkg_resources")
def mock_resource_filename(pkg_name, resource_name):
    spec = importlib.util.find_spec(pkg_name)
    return os.path.join(os.path.dirname(spec.origin), resource_name)
mock_pkg.resource_filename = mock_resource_filename
sys.modules['pkg_resources'] = mock_pkg 

import face_recognition
import cv2

def carregar_dados(dataset_path="dataset"):
    known_encodings = []
    known_names = []
    
    if not os.path.exists(dataset_path):
        return known_encodings, known_names

    for nome in os.listdir(dataset_path):
        caminho_pessoa = os.path.join(dataset_path, nome)
        if not os.path.isdir(caminho_pessoa):
            continue
            
        encodings_pessoa = []
        for arquivo in os.listdir(caminho_pessoa):
            if arquivo.endswith(('.jpg', '.png')):
                img_path = os.path.join(caminho_pessoa, arquivo)
                img = face_recognition.load_image_file(img_path)
                encodes = face_recognition.face_encodings(img)
                if len(encodes) > 0:
                    encodings_pessoa.append(encodes[0])
        
        if len(encodings_pessoa) > 0:
            vetor_medio = np.mean(encodings_pessoa, axis=0)
            known_encodings.append(vetor_medio)
            known_names.append(nome)
            
    return known_encodings, known_names

print("[INFO] A processar as biometrias guardadas...")
vetores, nomes = carregar_dados()

if len(vetores) == 0:
    print("[ERRO] Nenhum utilizador encontrado. Execute o script de cadastro primeiro.")
    sys.exit()

print("[INFO] A iniciar câmara. Pressione 'q' para sair.")
cap = cv2.VideoCapture(0)
TOLERANCIA = 0.5 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        top *= 4; right *= 4; bottom *= 4; left *= 4
        distancias = face_recognition.face_distance(vetores, face_encoding)
        
        nome_exibicao = "Nao cadastrado"
        cor_caixa = (0, 0, 255) 

        if len(distancias) > 0:
            melhor_indice = np.argmin(distancias)
            menor_distancia = distancias[melhor_indice]
            
            if menor_distancia <= TOLERANCIA:
                nome_exibicao = f"{nomes[melhor_indice]}"
                cor_caixa = (0, 255, 0) 

        cv2.rectangle(frame, (left, top), (right, bottom), cor_caixa, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), cor_caixa, cv2.FILLED)
        cv2.putText(frame, nome_exibicao, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Reconhecimento em Tempo Real", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()