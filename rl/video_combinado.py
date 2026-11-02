"""Video com a politica de caminhada e a de giro se alternando (como no programa).

    python rl/video_combinado.py <execucao caminhada> <execucao giro> [--saida arquivo.mp4]
"""
import argparse
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, ".."))
from controle.sim_mujoco import SimMuJoCo  # noqa: E402

# (inicio s, texto, frente m/s, giro rad/s)
ROTEIRO = [(0, "andando", 0.04, 0.0), (4, "girando para a esquerda", 0.0, 0.5), (9, "andando", 0.04, 0.0),
           (13, "girando para a direita", 0.0, -0.5), (18, "andando", 0.04, 0.0), (22, "fim", 0.0, 0.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("caminhada")
    ap.add_argument("giro")
    ap.add_argument("--saida")
    args = ap.parse_args()
    from PIL import Image, ImageDraw, ImageFont
    import imageio.v2 as imageio

    sim = SimMuJoCo(video=True)
    sim.carregar(os.path.join(args.caminhada, "politica.npz"))
    sim.carregar_giro(os.path.join(args.giro, "politica.npz"))
    try:
        fonte, menor = ImageFont.truetype("arial.ttf", 18), ImageFont.truetype("arial.ttf", 14)
    except OSError:
        fonte = menor = ImageFont.load_default()
    quadros = []
    rotulo = f"{os.path.basename(args.caminhada.rstrip(os.sep))} + {os.path.basename(args.giro.rstrip(os.sep))}"
    while sim.t < ROTEIRO[-1][0] and not sim.caiu:
        _, texto, vx, w = next(r for r in reversed(ROTEIRO) if sim.t >= r[0])
        sim.configurar(vx=vx, giro=w)
        sim.passo()
        img = Image.fromarray(sim.env.render())
        d = ImageDraw.Draw(img)
        info = sim.estado()["info"]
        d.text((14, 12), rotulo, fill=(235, 238, 242), font=fonte)
        d.text((14, 38), f"t = {sim.t:4.1f} s   {texto}", fill=(170, 178, 190), font=menor)
        d.text((14, 58), f"politica: {info['ativa']}   v = {info['vx'] * 100:4.1f} cm/s   giro {np.degrees(info['giro']):4.0f} graus/s   rumo {info['rumo']:4.0f} graus",
               fill=(255, 190, 80), font=menor)
        quadros.append(np.asarray(img))
    saida = args.saida or os.path.join(args.giro, "caminhar_e_girar.mp4")
    imageio.mimsave(saida, quadros, fps=round(1 / sim.env.dt))
    print(("CAIU " if sim.caiu else "") + f"{sim.t:.1f} s, rumo final {info['rumo']:.0f} graus")
    print(saida)


if __name__ == "__main__":
    main()
