"""Simulacao MuJoCo (robo livre, com chao) dentro do programa, guiada por politicas de RL.

Usa o mesmo ambiente do treino (rl/ambiente.py) e as politicas exportadas em .npz (numpy),
entao nao precisa de PyTorch. O servidor chama passo() a 50 Hz e manda pose + juntas ao visualizador.

Duas politicas: a de caminhada e, opcional, uma so de giro (rl/configs/giro.json). Quando o
comando e so girar, a de giro assume, com o filtro, a escala da acao e a passada do treino dela.

Sem politica, passo(q_alvo) leva as juntas aos angulos pedidos (poses, animacoes, sliders) e a
fisica decide o resto: o robo pode se equilibrar, escorregar ou cair.
"""
import json
import os
import sys

import mujoco
import numpy as np

from .modelo import RAIZ
from .politica_rl import PoliticaNumpy

sys.path.insert(0, os.path.join(RAIZ, "rl"))
from ambiente import JUNTAS, HumanoideAndar  # noqa: E402

AJUSTES_DA_POLITICA = ("escala_acao", "filtro_acao", "passada", "pose_padrao")


def _ler_config(caminho_npz):
    arq = os.path.join(os.path.dirname(caminho_npz), "config.json") if caminho_npz else ""
    if os.path.exists(arq):   # o ambiente precisa ser o mesmo do treino
        with open(arq, encoding="utf-8") as fp:
            return json.load(fp)
    return {}


class SimMuJoCo:
    def __init__(self, video=False):
        self.video = video          # True: modelo com as malhas do CAD e render rgb_array (gravar video)
        self.politica = None
        self.politica_giro = None
        self.cfg_politica = {}     # config.json do treino da politica de caminhada
        self.cfg_giro = {}
        self.cmd = np.array([0.04, 0.0, 0.0])   # frente (m/s), lado (m/s), giro (rad/s)
        self.massa = 1.0
        self.empurroes = False
        self.ativa = "caminhada"
        self.poses = False   # True: o servidor manda as juntas (poses); reinicia sem ruido
        self._criar()

    def _criar(self):
        r = {"massa": [self.massa, self.massa], "massa_elo": [1, 1], "carga": [0, 0], "deslocar_com": 0.0,
             "forca_servo": [1, 1], "atrito": [0.9, 0.9]}
        cfg = {k: v for k, v in self.cfg_politica.items() if k not in ("randomizar", "empurrao", "duracao", "comando")}
        omni = bool(self.cfg_politica.get("comando", {}).get("omni"))
        cfg.update({"randomizar": r, "empurrao": None if not self.empurroes else {
            "intervalo": [2.0, 4.0], "duracao": 0.1, "acel": [0.5, 2.5]}, "duracao": 1e9,
            "comando": {"omni": omni, "trocar": [1e9, 1e9]}})   # o comando vem do programa
        env = HumanoideAndar(cfg, render_mode="rgb_array" if self.video else None, visual=self.video)
        env.reset()   # pronto antes de trocar: o laco do servidor chama passo() de outra thread
        self.env = env
        self.ativa = "caminhada"
        self._base = {k: self.env.cfg[k] for k in AJUSTES_DA_POLITICA}
        self.reiniciar()

    @property
    def omni(self):
        return self.env.omni

    def configurar(self, vx=None, massa=None, empurroes=None, vy=None, giro=None):
        """Comando (vx, vy em m/s, giro em rad/s); vy e giro so tem efeito em politicas omni (v4+)."""
        for i, (v, lim) in enumerate(((vx, 0.2), (vy, 0.1), (giro, 1.5))):
            if v is not None:
                self.cmd[i] = float(np.clip(v, -lim, lim))
        self.env.cmd[:] = self.cmd
        recriar = False
        if massa is not None and abs(float(massa) - self.massa) > 1e-6:
            self.massa, recriar = float(np.clip(massa, 0.3, 3.0)), True
        if empurroes is not None and bool(empurroes) != self.empurroes:
            self.empurroes, recriar = bool(empurroes), True
        if recriar:
            self._criar()

    def carregar(self, caminho):
        self.politica = PoliticaNumpy(caminho) if caminho else None
        self.cfg_politica = _ler_config(caminho)
        self._criar()

    def carregar_giro(self, caminho):
        self.politica_giro = PoliticaNumpy(caminho) if caminho else None
        self.cfg_giro = _ler_config(caminho)

    def reiniciar(self):
        self.obs, _ = self.env.reset(seed=int(np.random.randint(1 << 30)))
        if self.poses:
            self._pose_exata()
        self.env.cmd[:] = self.cmd
        self.obs = self.env._obs()
        self.inicio = self.env.data.qpos[:2].copy()
        self.caiu = False
        self.toque = None
        self.t = 0.0
        self.reinicios = getattr(self, "reinicios", 0) + 1   # o servidor ressincroniza as poses

    def _pose_exata(self):
        """Reset do treino tem ruido de alguns graus nas juntas; parado, sem politica, isso derruba o robo."""
        e = self.env
        m, d = e.model, e.data
        d.qpos[e.qadr] = np.clip(e.q0, e.ctrl_lim[:, 0], e.ctrl_lim[:, 1])
        d.qvel[:] = 0.0
        d.ctrl[e.atuador] = d.qpos[e.qadr]
        mujoco.mj_forward(m, d)
        fundo = min(d.geom_xpos[g][2] - m.geom_size[g][2] for g in e.geom_pes)
        d.qpos[2] -= fundo - 0.0005
        mujoco.mj_forward(m, d)
        self.inicio = d.qpos[:2].copy()

    def empurrar(self, forca_rel=1.5):
        """Empurrao para frente/lado aleatorio, proporcional ao peso (m/s2 * massa)."""
        e = self.env
        ang = np.random.uniform(0, 2 * np.pi)
        e.forca_empurrao = e.massa_total * forca_rel * np.array([np.cos(ang), np.sin(ang), 0.0])
        e.fim_empurrao = e.data.time + 0.1

    def empurrar_em(self, corpo, ponto, direcao, forca=1.5, duracao=0.1):
        """Empurrao com o mouse: `forca` N na `direcao` (mundo), aplicada no `ponto` (mundo) do elo `corpo`.
        O ponto fica preso ao elo enquanto dura (o robo pode girar no meio do empurrao)."""
        e = self.env
        bid = mujoco.mj_name2id(e.model, mujoco.mjtObj.mjOBJ_BODY, corpo)
        if bid < 0:
            bid = e.torso
        R, p = e.data.xmat[bid].reshape(3, 3), e.data.xpos[bid]
        d = np.asarray(direcao, float)
        d = d / (np.linalg.norm(d) or 1.0)
        self.toque = (bid, R.T @ (np.asarray(ponto, float) - p), np.clip(forca, 0, 20) * d, e.data.time + duracao)

    def _aplicar_toque(self):
        d = self.env.data
        d.qfrc_applied[:] = 0.0   # o ambiente nao usa qfrc_applied: fica so para o empurrao do mouse
        if self.toque is None:
            return
        bid, local, f, fim = self.toque
        if d.time >= fim:
            self.toque = None
            return
        ponto = d.xpos[bid] + d.xmat[bid].reshape(3, 3) @ local
        mujoco.mj_applyFT(self.env.model, d, f, np.zeros(3), ponto, bid, d.qfrc_applied)

    def _escolher(self):
        """Politica de giro quando o comando e so girar; senao a de caminhada."""
        so_giro = abs(self.cmd[2]) > 0.05 and abs(self.cmd[0]) < 0.01 and abs(self.cmd[1]) < 0.005
        nome = "giro" if self.politica_giro and so_giro else "caminhada"
        if nome != self.ativa:
            self.ativa = nome
            cfg = self.cfg_giro if nome == "giro" else self.cfg_politica
            for k in AJUSTES_DA_POLITICA:   # filtro/escala/passada do treino da politica ativa
                self.env.cfg[k] = cfg.get(k, self._base[k])
        return self.politica_giro if nome == "giro" else self.politica

    def passo(self, q_alvo=None):
        """q_alvo (rad, ordem de JUNTAS): segue a pose em vez da politica e continua mesmo depois de cair."""
        self._aplicar_toque()
        if q_alvo is not None:
            self.ativa = "poses"
            self._passo_poses(np.asarray(q_alvo, float))
            return
        if self.ativa == "poses":   # voltou para a politica: o ambiente volta aos ajustes do treino
            self.ativa = None
        if self.caiu:
            return
        e = self.env
        pol = self._escolher()
        if pol is not None:
            e.omni = pol.camadas[0][0].shape[1] == 47   # observacao do tamanho que a politica espera
            acao = pol(e._obs())
        else:
            acao = np.zeros(len(JUNTAS))
        if not self.empurroes:   # sem empurroes automaticos o ambiente nao aplica a forca; aplica aqui
            e.data.xfrc_applied[e.torso, :3] = e.forca_empurrao if e.data.time < e.fim_empurrao else 0.0
        self.obs, _, self.caiu, _, self.info = e.step(acao)
        self.t += e.dt

    def _passo_poses(self, q_alvo):
        e = self.env
        e.data.ctrl[e.atuador] = np.clip(q_alvo, e.ctrl_lim[:, 0], e.ctrl_lim[:, 1])
        e.data.xfrc_applied[e.torso, :3] = e.forca_empurrao if e.data.time < e.fim_empurrao else 0.0
        mujoco.mj_step(e.model, e.data, nstep=e.n_sub)
        self.caiu = e._caiu()
        self.t += e.dt

    def estado(self):
        d = self.env.data
        q = d.qpos
        w, x, y, z = q[3:7]
        dist = q[:2] - self.inicio
        v = self.env._vel_corpo()
        R = self.env._rot_torso()
        nome = lambda p: os.path.basename(os.path.dirname(p.caminho)) if p else None  # noqa: E731
        return {
            "q": {n: float(np.degrees(q[a])) for n, a in zip(JUNTAS, self.env.qadr)},
            "base": {"p": q[:3].tolist(), "quat": [x, y, z, w]},
            "info": {
                "t": self.t, "caiu": bool(self.caiu), "vx": float(v[0]), "vy": float(v[1]), "giro": float(v[2]),
                "rumo": float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))), "omni": self.omni or bool(self.politica_giro),
                "cmd": self.cmd.tolist(), "vx_cmd": float(self.cmd[0]),
                "dist": float(np.hypot(*dist)), "massa": float(self.env.massa_total),
                "politica": nome(self.politica), "passos_politica": self.politica.passos if self.politica else 0,
                "politica_giro": nome(self.politica_giro), "ativa": self.ativa,
            },
        }
