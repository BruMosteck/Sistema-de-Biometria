"""Atalho para a comparação M²HD do aplicativo, sem outro reconhecedor."""
import tkinter as tk
from app_biometria import BiometriaApp


if __name__ == "__main__":
    root = tk.Tk()
    app = BiometriaApp(root)
    app.iniciar_verificacao()
    root.mainloop()
    app.encerrar_camera()
