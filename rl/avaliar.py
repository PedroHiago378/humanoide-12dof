"""Roda uma politica treinada no MuJoCo: janela 3D, video ou so estatisticas.

    python rl/avaliar.py rl/execucoes/<nome>/modelo_final.zip              # janela do MuJoCo
    python rl/avaliar.py <modelo.zip> --video caminhada.mp4 --segundos 10
    python rl/avaliar.py <modelo.zip> --episodios 20 --sem-janela          # so numeros
    python rl/avaliar.py <modelo.zip> --massa 1.3 --vx 0.08                # robo 30% mais pesado
    python rl/avaliar.py --aleatorio                                       # sem politica (teste do modelo)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ambiente import HumanoideAndar  # noqa: E402


def carregar(caminho, env):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    norm = VecNormalize.load(os.path.join(os.path.dirname(caminho), "vecnormalize.pkl"), DummyVecEnv([lambda: env]))
    norm.training, norm.norm_reward = False, False
    modelo = PPO.load(caminho, device="cpu")
    return lambda obs: modelo.predict(norm.normalize_obs(obs), deterministic=True)[0]


# (inicio s, texto, frente m/s, lado m/s, giro rad/s)
ROTEIRO = [(0, "frente", 0.04, 0.0, 0.0), (4, "virando para a esquerda", 0.015, 0.0, 0.35),
           (8, "virando para a direita", 0.015, 0.0, -0.35), (12, "de lado para a esquerda", 0.0, 0.015, 0.0),
           (16, "de lado para a direita", 0.0, -0.015, 0.0), (20, "parado", 0.0, 0.0, 0.0)]


# politica so de giro (rl/configs/giro.json)
ROTEIRO_GIRO = [(0, "girando para a esquerda", 0.0, 0.0, 0.5), (5, "girando para a direita", 0.0, 0.0, -0.5),
                (10, "girando rapido para a esquerda", 0.0, 0.0, 0.8), (14, "girando rapido para a direita", 0.0, 0.0, -0.8),
                (18, "parado", 0.0, 0.0, 0.0), (20, "parado", 0.0, 0.0, 0.0)]


def roteiro_de(env):
    c = env.cfg["comando"]
    return ROTEIRO_GIRO if c.get("vx") == [0.0, 0.0] and c.get("vy") == [0.0, 0.0] else ROTEIRO


def aplicar_roteiro(env, t):
    _, _, vx, vy, w = next(r for r in reversed(roteiro_de(env)) if t >= r[0])
    env.cmd[:] = (vx, vy, w)
    return env._obs()


def legendar(quadro, texto, t, env):
    """Escreve a legenda, o tempo e a velocidade no quadro do video."""
    if not texto:
        return quadro
    from PIL import Image, ImageDraw, ImageFont
    img = Image.fromarray(quadro)
    d = ImageDraw.Draw(img)
    try:
        fonte = ImageFont.truetype("arial.ttf", 18)
        menor = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        fonte = menor = ImageFont.load_default()
    d.text((14, 12), texto, fill=(235, 238, 242), font=fonte)
    v = env._vel_corpo()
    d.text((14, 38), f"t = {t:4.1f} s   v = {v[0] * 100:4.1f} cm/s", fill=(170, 178, 190), font=menor)
    if env.omni:
        acao = next(r[1] for r in reversed(roteiro_de(env)) if t >= r[0])
        d.text((14, 58), f"{acao}   lado {v[1] * 100:4.1f} cm/s   giro {np.degrees(v[2]):4.0f} graus/s",
               fill=(255, 190, 80), font=menor)
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modelo", nargs="?")
    ap.add_argument("--aleatorio", action="store_true", help="acoes zero (so a pose padrao), sem politica")
    ap.add_argument("--episodios", type=int, default=3)
    ap.add_argument("--segundos", type=float, default=20.0)
    ap.add_argument("--video")
    ap.add_argument("--sem-janela", action="store_true")
    ap.add_argument("--massa", type=float, help="escala fixa da massa (sem randomizacao)")
    ap.add_argument("--vx", type=float, help="comando de velocidade fixo (m/s)")
    ap.add_argument("--sem-empurrao", action="store_true")
    ap.add_argument("--legenda", help="texto escrito no canto do video")
    ap.add_argument("--roteiro", action="store_true",
                    help="politica omni: frente, vira para a esquerda, vira para a direita, anda de lado (4 s cada)")
    args = ap.parse_args()

    config = {}
    arq_cfg = os.path.join(os.path.dirname(args.modelo), "config.json") if args.modelo else ""
    if os.path.exists(arq_cfg):   # mesmo ambiente do treino (filtro, escala da acao, passada...)
        with open(arq_cfg, encoding="utf-8") as fp:
            config = json.load(fp)
    config["duracao"] = args.segundos
    if args.massa:
        config["randomizar"] = {"massa": [args.massa, args.massa], "massa_elo": [1, 1], "carga": [0, 0],
                                "deslocar_com": 0.0, "forca_servo": [1, 1], "atrito": [0.9, 0.9]}
    if args.sem_empurrao:
        config["empurrao"] = None
    if args.roteiro:
        config.setdefault("comando", {})["trocar"] = [1e9, 1e9]   # o comando vem do roteiro
    modo = "rgb_array" if args.video else (None if args.sem_janela else "human")
    env = HumanoideAndar(config, render_mode=modo, visual=True)
    politica = (lambda obs: np.zeros(12)) if args.aleatorio or not args.modelo else carregar(args.modelo, env)

    quadros = []
    for ep in range(args.episodios):
        obs, _ = env.reset(seed=ep)
        if args.vx is not None:
            env.vx_cmd = args.vx
        total, passos, inicio = 0.0, 0, env.data.qpos[:2].copy()
        qd_ant, acel, var_alvo, alvo_ant = env.data.qvel[env.vadr].copy(), 0.0, 0.0, env.data.ctrl[env.atuador].copy()
        while True:
            t0 = time.perf_counter()
            if args.roteiro:
                obs = aplicar_roteiro(env, passos * env.dt) if env.omni else obs
            obs, r, caiu, trunc, info = env.step(politica(obs))
            total += r
            passos += 1
            qd = env.data.qvel[env.vadr]
            acel += np.mean(np.abs(qd - qd_ant)) / env.dt          # rad/s2 medio
            var_alvo += np.degrees(np.mean(np.abs(env.data.ctrl[env.atuador] - alvo_ant)))
            qd_ant, alvo_ant = qd.copy(), env.data.ctrl[env.atuador].copy()
            if args.video:
                quadros.append(legendar(env.render(), args.legenda, passos * env.dt, env))
            elif modo == "human":
                time.sleep(max(0.0, env.dt - (time.perf_counter() - t0)))
            if caiu or trunc:
                break
        dist = env.data.qpos[0] - inicio[0]
        print(f"episodio {ep}: {passos * env.dt:5.1f} s {'CAIU' if caiu else 'em pe'}, "
              f"andou {dist * 100:5.1f} cm (cmd {env.vx_cmd * 100:.1f} cm/s, media {dist / (passos * env.dt) * 100:.1f} cm/s), "
              f"massa {info['massa'] * 1000:.0f} g, recompensa {total:.1f}, "
              f"fluidez: acel. juntas {acel / passos:.0f} rad/s2, variacao do alvo {var_alvo / passos:.2f} graus/quadro")
        if args.video:
            break
    env.close()
    if args.video:
        import imageio.v2 as imageio
        imageio.mimsave(args.video, quadros, fps=round(1 / env.dt))
        print("Video:", args.video)


if __name__ == "__main__":
    main()
