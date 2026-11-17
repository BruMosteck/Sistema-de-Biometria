"""Atalho para o mesmo cadastro local do aplicativo, sem serviços externos."""
import tkinter as tk
from app_biometria import BiometriaApp


if __name__ == "__main__":
    root = tk.Tk()
    app = BiometriaApp(root)
    app.iniciar_cadastro()
    root.mainloop()
    app.encerrar_camera()
