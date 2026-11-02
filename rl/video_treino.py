"""Grava um video da politica atual de um treino (mesmo em andamento).

    python rl/video_treino.py <pasta da execucao> [--segundos 15]

Copia modelo_atual.zip (ou modelo_final.zip) + vecnormalize.pkl + config.json para uma pasta
temporaria (o treino pode estar gravando por cima) e salva <execucao>_<passos>M.mp4 na pasta da execucao.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

AQUI = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pasta")
    ap.add_argument("--segundos", type=float, default=15.0)
    ap.add_argument("--vx", type=float, default=0.04)
    args = ap.parse_args()
    pasta = os.path.abspath(args.pasta)
    nome = os.path.basename(pasta)
    modelo = next((os.path.join(pasta, m) for m in ("modelo_final.zip", "modelo_atual.zip")
                   if os.path.exists(os.path.join(pasta, m))), None)
    if modelo is None:
        sys.exit("execucao sem modelo salvo ainda")
    with tempfile.TemporaryDirectory() as tmp:
        for arq in (modelo, os.path.join(pasta, "vecnormalize.pkl"), os.path.join(pasta, "config.json")):
            shutil.copy(arq, tmp)
        import zipfile
        with zipfile.ZipFile(os.path.join(tmp, os.path.basename(modelo))) as z:
            dados = json.loads(z.read("data"))
        passos = int(dados.get("num_timesteps", 0))
        with open(os.path.join(pasta, "config.json"), encoding="utf-8") as fp:
            omni = json.load(fp).get("comando", {}).get("omni", False)
        saida = os.path.join(pasta, f"{nome}_{passos / 1e6:04.1f}M.mp4")
        subprocess.run([sys.executable, os.path.join(AQUI, "avaliar.py"), os.path.join(tmp, os.path.basename(modelo)),
                        "--video", saida, "--segundos", str(max(args.segundos, 22.0) if omni else args.segundos),
                        "--vx", str(args.vx), *(["--roteiro"] if omni else []),
                        "--massa", "1.0", "--sem-empurrao", "--legenda", f"{nome} · {passos / 1e6:.1f} M passos"],
                       check=True)
    print(saida)


if __name__ == "__main__":
    main()
