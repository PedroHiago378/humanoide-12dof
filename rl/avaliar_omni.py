"""Mede quanto uma politica omni segue cada comando do roteiro (frente, virar, de lado).

    python rl/avaliar_omni.py <pasta da execucao ou politica.npz> [--episodios 3]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ambiente import HumanoideAndar  # noqa: E402
from avaliar import roteiro_de  # noqa: E402
from controle.politica_rl import PoliticaNumpy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("caminho")
    ap.add_argument("--episodios", type=int, default=3)
    args = ap.parse_args()
    npz = args.caminho if args.caminho.endswith(".npz") else os.path.join(args.caminho, "politica.npz")
    with open(os.path.join(os.path.dirname(npz), "config.json"), encoding="utf-8") as fp:
        cfg = json.load(fp)
    cfg.update({"empurrao": None,
                "randomizar": {"massa": [1, 1], "massa_elo": [1, 1], "carga": [0, 0], "deslocar_com": 0.0,
                               "forca_servo": [1, 1], "atrito": [0.9, 0.9]}})
    cfg["comando"]["trocar"] = [1e9, 1e9]
    env, pol = HumanoideAndar(cfg), PoliticaNumpy(npz)
    ROTEIRO = roteiro_de(env)
    env.max_passos = int(ROTEIRO[-1][0] / env.dt)
    medidas = {r[1]: [] for r in ROTEIRO[:-1]}
    quedas = 0
    for ep in range(args.episodios):
        obs, _ = env.reset(seed=ep)
        while True:
            t = env.data.time
            seg = next(r for r in reversed(ROTEIRO) if t >= r[0])
            env.cmd[:] = seg[2:]
            obs = env._obs()
            obs, _, caiu, trunc, _ = env.step(pol(obs))
            if t - seg[0] > 1.0 and seg[1] in medidas:   # ignora o primeiro segundo de cada trecho
                medidas[seg[1]].append(env._vel_corpo())
            if caiu or trunc:
                quedas += caiu
                break
    print(f"{os.path.basename(os.path.dirname(npz))} ({pol.passos / 1e6:.1f} M passos), quedas: {quedas}/{args.episodios}")
    for r in ROTEIRO[:-1]:
        v = np.mean(medidas[r[1]], axis=0) if medidas[r[1]] else np.zeros(3)
        print(f"  {r[1]:26s} pedido: frente {r[2] * 100:4.1f} lado {r[3] * 100:4.1f} cm/s giro {np.degrees(r[4]):4.0f} graus/s"
              f" | medido: frente {v[0] * 100:4.1f} lado {v[1] * 100:4.1f} cm/s giro {np.degrees(v[2]):4.0f} graus/s")


if __name__ == "__main__":
    main()
