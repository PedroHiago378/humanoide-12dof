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
    # simulacao MuJoCo e treino (aba Simulacao e treino); o treino em si roda no Python do PC
    ("rl/ambiente.py", "rl"),
    ("rl/treinar.py", "rl"),
    ("rl/exportar.py", "rl"),
    ("rl/avaliar.py", "rl"),
    ("modelo/mujoco/robo_treino.xml", "modelo/mujoco"),
    ("modelo/mujoco/colisao", "modelo/mujoco/colisao"),
]

cmd = [
    sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onefile", "--windowed",
    "--name", "Humanoide",
    "--distpath", os.path.join(RAIZ, "dist"),
    "--workpath", os.path.join(RAIZ, "build"),
    "--specpath", os.path.join(RAIZ, "build"),
    "--hidden-import", "serial.tools.list_ports",
    # importados so quando a aba de simulacao e usada (rl/ambiente.py e carregado do disco)
    "--hidden-import", "controle.sim_mujoco", "--hidden-import", "controle.rl_servico",
    "--hidden-import", "controle.politica_rl", "--hidden-import", "gymnasium", "--hidden-import", "gymnasium.spaces",
    "--collect-all", "mujoco",
    "--exclude-module", "tkinter", "--exclude-module", "matplotlib", "--exclude-module", "scipy",
    "--exclude-module", "OCP", "--exclude-module", "cadquery", "--exclude-module", "trimesh",
    "--exclude-module", "torch", "--exclude-module", "stable_baselines3", "--exclude-module", "tensorboard",
    "--exclude-module", "pandas", "--exclude-module", "cv2", "--exclude-module", "imageio",
    "--exclude-module", "imageio_ffmpeg", "--exclude-module", "pygame", "--exclude-module", "PIL",
    "--exclude-module", "lxml", "--exclude-module", "pygments", "--exclude-module", "IPython",
]
if os.path.exists(os.path.join(RAIZ, "ferramentas", "icone.ico")):
    cmd += ["--icon", os.path.join(RAIZ, "ferramentas", "icone.ico")]
for origem, destino in dados:
    cmd += ["--add-data", f"{os.path.join(RAIZ, origem)}{SEP}{destino}"]
cmd.append(os.path.join(RAIZ, "app_humanoide.py"))

subprocess.run(cmd, check=True, cwd=RAIZ)
print("Pronto:", os.path.join(RAIZ, "dist", "Humanoide.exe"))
