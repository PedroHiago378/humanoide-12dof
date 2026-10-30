"""Gera dist/Humanoide.exe (PyInstaller, arquivo unico, sem console).

    python ferramentas/gerar_exe.py
"""
import os
import subprocess
import sys

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEP = os.pathsep

dados = [
    ("visualizador", "visualizador"),
    ("modelo/robo.json", "modelo"),
    ("modelo/robo.glb", "modelo"),
    ("config/poses.json", "config"),
]

cmd = [
    sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onefile", "--windowed",
    "--name", "Humanoide",
    "--distpath", os.path.join(RAIZ, "dist"),
    "--workpath", os.path.join(RAIZ, "build"),
    "--specpath", os.path.join(RAIZ, "build"),
    "--hidden-import", "serial.tools.list_ports",
    "--exclude-module", "tkinter", "--exclude-module", "matplotlib", "--exclude-module", "scipy",
    "--exclude-module", "OCP", "--exclude-module", "cadquery", "--exclude-module", "trimesh",
]
if os.path.exists(os.path.join(RAIZ, "ferramentas", "icone.ico")):
    cmd += ["--icon", os.path.join(RAIZ, "ferramentas", "icone.ico")]
for origem, destino in dados:
    cmd += ["--add-data", f"{os.path.join(RAIZ, origem)}{SEP}{destino}"]
cmd.append(os.path.join(RAIZ, "app_humanoide.py"))

subprocess.run(cmd, check=True, cwd=RAIZ)
print("Pronto:", os.path.join(RAIZ, "dist", "Humanoide.exe"))
