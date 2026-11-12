import sys
import os
import types
import importlib.util

# Monkey Patch para corrigir o erro 'pkg_resources' no Python 3.14+
mock_pkg = types.ModuleType("pkg_resources")
def mock_resource_filename(pkg_name, resource_name):
    spec = importlib.util.find_spec(pkg_name)
    return os.path.join(os.path.dirname(spec.origin), resource_name)
mock_pkg.resource_filename = mock_resource_filename
sys.modules['pkg_resources'] = mock_pkg 

import cv2
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

print("[INFO] Autenticando no Google Drive...")
gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

def enviar_para_nuvem(caminho_local, nome_arquivo_nuvem):
    arquivo = drive.CreateFile({'title': nome_arquivo_nuvem})
    arquivo.SetContentFile(caminho_local)
    arquivo.Upload()
    print(f"[INFO] Arquivo {nome_arquivo_nuvem} salvo na nuvem com sucesso!")

nome = input("Digite o nome do usuário para cadastro: ").strip().title()
diretorio = f"dataset/{nome}"

if not os.path.exists(diretorio):
    os.makedirs(diretorio)

cap = cv2.VideoCapture(0)
fotos_tiradas = 0

print("[INFO] Pressione 'c' para capturar a foto (3 necessárias). Pressione 'q' para sair.")

while fotos_tiradas < 3:
    ret, frame = cap.read()
    if not ret:
        print("[ERRO] Falha ao capturar a câmera.")
        break

    frame = cv2.flip(frame, 1)
    frame_display = frame.copy()

    h, w = frame_display.shape[:2]
    cv2.ellipse(frame_display, (w//2, h//2), (100, 150), 0, 0, 360, (255, 255, 0), 2)
    cv2.putText(frame_display, f"Fotos: {fotos_tiradas}/3", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Cadastro de Biometria", frame_display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        caminho_foto = os.path.join(diretorio, f"foto_{fotos_tiradas+1}.jpg")
        cv2.imwrite(caminho_foto, frame)
        nome_arquivo_nuvem = f"{nome}_foto_{fotos_tiradas+1}.jpg"
        enviar_para_nuvem(caminho_foto, nome_arquivo_nuvem)
        fotos_tiradas += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if fotos_tiradas == 3:
    print(f"[INFO] Cadastro do usuário {nome} concluído com sucesso!")