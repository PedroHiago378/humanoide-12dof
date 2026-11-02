"""Ambiente Gymnasium: humanoide de 12 juntas aprendendo a andar no MuJoCo.

Observacao (so o que o robo real tem: IMU + servos + relogio):
  gravidade no frame do torso (3), giroscopio (3), juntas - pose padrao (12), velocidade das
  juntas (12), acao anterior (12), comando de velocidade (1), fase da passada sin/cos (2) = 45
  Com "comando": {"omni": true} o comando vira (frente, lado, giro) e a observacao tem 47 valores:
  o robo aprende a andar em qualquer direcao e a virar para os dois lados.
Acao: 12 valores em [-1, 1] -> alvo dos servos = pose padrao + acao * escala (50 Hz, como na ESP32)

Equilibrio independente do peso:
  - a cada episodio a massa do robo muda (escala geral, por elo, carga extra no torso com o centro
    de massa deslocado), assim como a forca dos servos e o atrito; a politica nao recebe a massa,
    entao precisa equilibrar qualquer um desses robos;
  - as recompensas usam grandezas normalizadas (torque / torque maximo, centro de massa em relacao
    ao pe, inclinacao), que nao dependem de quanto o robo pesa;
  - empurroes com forca proporcional ao peso treinam a recuperacao do equilibrio.
Todos os pesos das recompensas ficam em PADRAO["recompensas"] (ou no --config do treino).
"""
import copy
import os

import gymnasium as gym
import mujoco
import numpy as np

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
XML_TREINO = os.path.join(RAIZ, "modelo", "mujoco", "robo_treino.xml")
XML_VISUAL = os.path.join(RAIZ, "modelo", "mujoco", "robo.xml")

JUNTAS = ["quadril_roll_L", "quadril_pitch_L", "joelho_L", "tornozelo_L",
          "quadril_roll_R", "quadril_pitch_R", "joelho_R", "tornozelo_R",
          "ombro_L", "cotovelo_L", "ombro_R", "cotovelo_R"]

PADRAO = {
    "dt_controle": 0.02,            # 50 Hz
    "duracao": 20.0,                # s por episodio
    "escala_acao": 0.5,             # rad: alcance da acao em torno da pose padrao
    "filtro_acao": 0.0,             # passa-baixa na acao: 0 = sem filtro, 0.5 = metade vem do quadro anterior
    "pose_padrao": {                # graus: joelho levemente dobrado, bracos baixos
        "quadril_pitch_L": -15, "joelho_L": 30, "tornozelo_L": -15,
        "quadril_pitch_R": -15, "joelho_R": 30, "tornozelo_R": -15,
        "cotovelo_L": -80, "cotovelo_R": 80,
    },
    "comando": {
        "vx_max": 0.12, "prob_parado": 0.2,   # m/s (modo so para frente)
        "omni": False,                          # True: comando (vx, vy, giro), observacao com 47 valores
        "vx": [-0.03, 0.05], "vy": [-0.02, 0.02], "giro": [-0.6, 0.6],   # m/s, m/s, rad/s
        "prob_zero": 0.35,                      # chance de cada parte do comando ser zero
        "trocar": [3.0, 6.0],                   # s entre trocas de comando no meio do episodio
    },
    "passada": {"periodo": 0.6, "altura_pe": 0.012},   # s, m
    "ruido_obs": {"giro": 0.05, "gravidade": 0.03, "q": 0.01, "qd": 0.3},
    "randomizar": {
        "massa": [0.75, 1.35],      # escala da massa total
        "massa_elo": [0.9, 1.1],    # escala extra por elo
        "carga": [0.0, 0.12],       # kg somados ao torso
        "deslocar_com": 0.01,       # m, deslocamento do centro de massa do torso
        "forca_servo": [0.8, 1.2],
        "atrito": [0.5, 1.2],
    },
    "empurrao": {"intervalo": [2.0, 4.0], "duracao": 0.1, "acel": [0.5, 2.5]},   # acel em m/s2 * massa
    "queda": {"altura": 0.075, "inclinacao": 55.0},     # m do torso, graus
    "recompensas": {
        "velocidade": 1.5,          # seguir o comando para frente
        "lateral": 0.5,             # nao andar de lado nem girar (so no modo para frente)
        "giro": 0.0,                # seguir o giro pedido (modo omni)
        "direcao": 0.0,             # progresso na direcao pedida de lado/giro, mesmo parcial (modo omni)
        "postura": 1.0,             # torso reto (equilibrio)
        "altura": 0.5,              # torso na altura da pose padrao
        "centro_massa": 0.6,        # centro de massa sobre os pes (equilibrio)
        "fase": 0.4,                # contato dos pes seguindo a fase da passada
        "altura_pe": 0.3,           # pe de balanco levantado
        "vivo": 0.2,
        "torque": -0.05,            # (tau / tau_max)^2
        "suavidade": -0.05,         # variacao da acao
        "tremor": 0.0,              # segunda diferenca da acao (vai-e-volta rapido)
        "vel_juntas": 0.0,          # (velocidade da junta / velocidade maxima do servo)^2
        "acel_juntas": 0.0,         # aceleracao das juntas (normalizada por 200 rad/s2)
        "limite": -0.1,             # juntas encostadas no limite
        "escorregar": -0.1,         # pe em contato deslizando
        "queda": -5.0,
    },
}


def _mesclar(base, novo):
    for k, v in (novo or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict) and k != "pose_padrao":
            _mesclar(base[k], v)
        else:
            base[k] = v
    return base


class HumanoideAndar(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

    def __init__(self, config=None, render_mode=None, visual=False):
        self.cfg = _mesclar(copy.deepcopy(PADRAO), config)
        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_path(XML_VISUAL if visual or render_mode else XML_TREINO)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.n_sub = max(1, round(self.cfg["dt_controle"] / m.opt.timestep))
        self.dt = self.n_sub * m.opt.timestep
        self.max_passos = int(self.cfg["duracao"] / self.dt)

        self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in JUNTAS])
        self.vadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in JUNTAS])
        self.atuador = np.array([m.actuator(n).id for n in JUNTAS])
        self.ctrl_lim = m.actuator_ctrlrange[self.atuador].copy()
        self.tau_max = m.actuator_forcerange[self.atuador, 1].copy()
        self.torso = m.body("torso").id
        self.pes = [m.body(p).id for p in ("pe_L", "pe_R")]
        self.geom_pes = [m.geom(p).id for p in ("pe_L", "pe_R")]
        self.site_pes = [m.site(p).id for p in ("pe_L", "pe_R")]
        self.chao = m.geom("chao").id
        self.q0 = np.radians([self.cfg["pose_padrao"].get(n, 0.0) for n in JUNTAS])
        self.q0 = np.clip(self.q0, self.ctrl_lim[:, 0], self.ctrl_lim[:, 1])

        # valores nominais para a randomizacao
        self.massa0 = m.body_mass.copy()
        self.inercia0 = m.body_inertia.copy()
        self.ipos0 = m.body_ipos.copy()
        self.ganho0 = m.actuator_gainprm[:, 0].copy()
        self.vies0 = m.actuator_biasprm[:, :3].copy()
        self.forca0 = m.actuator_forcerange.copy()
        self.atrito0 = m.geom_friction[:, 0].copy()

        self.action_space = gym.spaces.Box(-1.0, 1.0, (len(JUNTAS),), np.float32)
        self.omni = bool(self.cfg["comando"].get("omni"))
        self.cmd = np.zeros(3)   # frente (m/s), lado (m/s), giro (rad/s)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (47 if self.omni else 45,), np.float32)
        self._renderer = None
        self._viewer = None

    # ------------------------------------------------------------------ randomizacao
    def _randomizar(self):
        m, r, rnd = self.model, self.cfg["randomizar"], self.np_random
        escala = rnd.uniform(*r["massa"]) * rnd.uniform(*r["massa_elo"], size=m.nbody)
        escala[0] = 1.0  # mundo
        m.body_mass[:] = self.massa0 * escala
        m.body_inertia[:] = self.inercia0 * escala[:, None]
        carga = rnd.uniform(*r["carga"])
        m.body_mass[self.torso] += carga
        m.body_ipos[:] = self.ipos0
        m.body_ipos[self.torso, :2] += rnd.uniform(-r["deslocar_com"], r["deslocar_com"], size=2)
        f = rnd.uniform(*r["forca_servo"], size=m.nu)
        m.actuator_gainprm[:, 0] = self.ganho0 * f
        m.actuator_biasprm[:, :3] = self.vies0 * f[:, None]
        m.actuator_forcerange[:] = self.forca0 * f[:, None]
        m.geom_friction[:, 0] = self.atrito0
        m.geom_friction[[self.chao, *self.geom_pes], 0] = rnd.uniform(*r["atrito"])
        self.tau_max = m.actuator_forcerange[self.atuador, 1].copy()
        self.massa_total = m.body_mass.sum()

    # ------------------------------------------------------------------ gym
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        m, d, rnd = self.model, self.data, self.np_random
        if self.cfg["randomizar"]:
            self._randomizar()
        else:
            self.massa_total = m.body_mass.sum()
        mujoco.mj_resetData(m, d)
        q = self.q0 + rnd.uniform(-0.05, 0.05, size=len(JUNTAS))
        d.qpos[self.qadr] = np.clip(q, self.ctrl_lim[:, 0], self.ctrl_lim[:, 1])
        d.ctrl[self.atuador] = d.qpos[self.qadr]
        # apoia os pes no chao
        mujoco.mj_forward(m, d)
        fundo = min(d.geom_xpos[g][2] - m.geom_size[g][2] for g in self.geom_pes)
        d.qpos[2] -= fundo - 0.0005
        mujoco.mj_forward(m, d)
        self.altura_alvo = d.xpos[self.torso][2]

        self._sortear_comando()
        self.t_troca = rnd.uniform(*self.cfg["comando"]["trocar"]) if self.omni else np.inf
        self.passos = 0
        self.acao_ant = np.zeros(len(JUNTAS))
        self.acao_ant2 = np.zeros(len(JUNTAS))
        self.acao_filtrada = np.zeros(len(JUNTAS))
        self.qd_ant = np.zeros(len(JUNTAS))
        self.t_empurrao = rnd.uniform(*self.cfg["empurrao"]["intervalo"]) if self.cfg["empurrao"] else np.inf
        self.forca_empurrao = np.zeros(3)
        self.fim_empurrao = -1.0
        return self._obs(), {}

    def step(self, acao):
        m, d = self.model, self.data
        acao = np.clip(np.asarray(acao, float), -1, 1)
        f = self.cfg["filtro_acao"]
        self.acao_filtrada = f * self.acao_filtrada + (1 - f) * acao
        alvo = np.clip(self.q0 + self.acao_filtrada * self.cfg["escala_acao"], self.ctrl_lim[:, 0], self.ctrl_lim[:, 1])
        d.ctrl[self.atuador] = alvo
        self._empurrar()
        mujoco.mj_step(m, d, nstep=self.n_sub)
        self.passos += 1
        if d.time >= self.t_troca:   # modo omni: muda o comando no meio do episodio
            self._sortear_comando()
            self.t_troca = d.time + self.np_random.uniform(*self.cfg["comando"]["trocar"])
        recompensa, partes = self._recompensa(acao)
        caiu = self._caiu()
        if caiu:
            recompensa += self.cfg["recompensas"]["queda"]
        self.acao_ant2, self.acao_ant = self.acao_ant, acao
        self.qd_ant = d.qvel[self.vadr].copy()
        truncado = self.passos >= self.max_passos
        info = {"recompensas": partes, "vx": self._vel_corpo()[0], "vx_cmd": self.vx_cmd,
                "massa": self.massa_total, "cmd": self.cmd.copy()}
        if self.render_mode == "human":
            self.render()
        return self._obs(), float(recompensa), caiu, truncado, info

    # ------------------------------------------------------------------ comando
    @property
    def vx_cmd(self):
        return self.cmd[0]

    @vx_cmd.setter
    def vx_cmd(self, valor):
        self.cmd[0] = valor

    def _sortear_comando(self):
        c, rnd = self.cfg["comando"], self.np_random
        if rnd.random() < c["prob_parado"]:
            self.cmd = np.zeros(3)
        elif not self.omni:
            self.cmd = np.array([rnd.uniform(0.02, c["vx_max"]), 0.0, 0.0])
        else:
            self.cmd = np.array([rnd.uniform(*c[k]) for k in ("vx", "vy", "giro")])
            self.cmd[rnd.random(3) < c["prob_zero"]] = 0.0

    # ------------------------------------------------------------------ partes
    def _empurrar(self):
        e, t = self.cfg["empurrao"], self.data.time
        if not e:
            return
        if t >= self.t_empurrao:
            ang = self.np_random.uniform(0, 2 * np.pi)
            a = self.np_random.uniform(*e["acel"])
            self.forca_empurrao = self.massa_total * a * np.array([np.cos(ang), np.sin(ang), 0.0])
            self.fim_empurrao = t + e["duracao"]
            self.t_empurrao = t + self.np_random.uniform(*e["intervalo"])
        self.data.xfrc_applied[self.torso, :3] = self.forca_empurrao if t < self.fim_empurrao else 0.0

    def _rot_torso(self):
        return self.data.xmat[self.torso].reshape(3, 3)

    def _gravidade(self):
        return self._rot_torso().T @ np.array([0.0, 0.0, -1.0])

    def _vel_corpo(self):
        """Velocidade linear do torso no frame de rumo (yaw) e taxa de giro em z."""
        R = self._rot_torso()
        yaw = np.arctan2(R[1, 0], R[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        v = self.data.qvel[:3]
        return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1], self.data.qvel[5]])

    def _fase(self):
        return (self.data.time / self.cfg["passada"]["periodo"]) % 1.0

    def _contatos(self):
        tocando = [False, False]
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            for k, g in enumerate(self.geom_pes):
                if (c.geom1 == g and c.geom2 == self.chao) or (c.geom2 == g and c.geom1 == self.chao):
                    tocando[k] = True
        return np.array(tocando)

    def _obs(self):
        d, ruido, rnd = self.data, self.cfg["ruido_obs"], self.np_random
        fase = 2 * np.pi * self._fase()
        giro = d.sensor("giro").data
        obs = np.concatenate([
            self._gravidade() + rnd.normal(0, ruido["gravidade"], 3),
            (giro + rnd.normal(0, ruido["giro"], 3)) * 0.25,
            d.qpos[self.qadr] - self.q0 + rnd.normal(0, ruido["q"], 12),
            (d.qvel[self.vadr] + rnd.normal(0, ruido["qd"], 12)) * 0.05,
            self.acao_ant,
            [self.cmd[0] * 5.0, self.cmd[1] * 5.0, self.cmd[2]] if self.omni else [self.cmd[0] * 5.0],
            [np.sin(fase), np.cos(fase)],
        ])
        return obs.astype(np.float32)

    def _caiu(self):
        q = self.cfg["queda"]
        altura = self.data.xpos[self.torso][2]
        inclinacao = np.degrees(np.arccos(np.clip(-self._gravidade()[2], -1, 1)))
        return bool(altura < q["altura"] or inclinacao > q["inclinacao"])

    def _recompensa(self, acao):
        m, d, w = self.model, self.data, self.cfg["recompensas"]
        v = self._vel_corpo()
        g = self._gravidade()
        contato = self._contatos()
        fase = self._fase()
        balanco = np.array([fase < 0.5, fase >= 0.5])        # pe esquerdo no ar na 1a metade
        if not self.cmd.any():
            balanco[:] = False                                # parado: os dois pes no chao

        # centro de massa sobre o apoio, em unidades do tamanho do pe (vale para qualquer massa)
        com = d.subtree_com[self.torso][:2]
        pes_xy = np.array([d.site_xpos[s][:2] for s in self.site_pes])
        apoio = pes_xy[contato].mean(axis=0) if contato.any() else pes_xy.mean(axis=0)
        dist_com = np.linalg.norm(com - apoio) / 0.03

        tau = d.actuator_force[self.atuador] / self.tau_max
        q = d.qpos[self.qadr]
        perto_limite = ((q - self.ctrl_lim[:, 0] < 0.08) | (self.ctrl_lim[:, 1] - q < 0.08)).mean()
        escorrega = 0.0
        vel6 = np.zeros(6)
        altura_pe = 0.0
        for k, s in enumerate(self.site_pes):
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, s, vel6, 0)
            if contato[k]:
                escorrega += np.linalg.norm(vel6[3:5])
            if balanco[k]:
                z = d.site_xpos[s][2] - m.site_size[s][2]
                altura_pe += np.exp(-((z - self.cfg["passada"]["altura_pe"]) / 0.006) ** 2)

        partes = {
            "velocidade": np.exp(-((v[0] - self.cmd[0]) ** 2 + (v[1] - self.cmd[1]) ** 2) / 0.04 ** 2) if self.omni
            else np.exp(-((v[0] - self.cmd[0]) / 0.04) ** 2),
            "lateral": 1.0 if self.omni else np.exp(-(v[1] / 0.04) ** 2 - (v[2] / 0.5) ** 2),
            "giro": np.exp(-((v[2] - self.cmd[2]) / 0.3) ** 2),
            # fracao do lado/giro pedido que ja acontece (-1..1): da gradiente mesmo longe do alvo
            "direcao": sum(np.clip(v[i] / self.cmd[i], -1.0, 1.0) for i in (1, 2) if abs(self.cmd[i]) > 1e-3),
            "postura": np.exp(-(g[0] ** 2 + g[1] ** 2) / 0.02),
            "altura": np.exp(-((d.xpos[self.torso][2] - self.altura_alvo) / 0.012) ** 2),
            "centro_massa": np.exp(-dist_com ** 2),
            "fase": (contato == ~balanco).mean(),
            "altura_pe": altura_pe / max(1, balanco.sum()) if balanco.any() else 0.0,
            "vivo": 1.0,
            "torque": np.mean(tau ** 2),
            "suavidade": np.mean((acao - self.acao_ant) ** 2),
            "tremor": np.mean((acao - 2 * self.acao_ant + self.acao_ant2) ** 2),
            "vel_juntas": np.mean((d.qvel[self.vadr] / 10.5) ** 2),
            "acel_juntas": np.mean(((d.qvel[self.vadr] - self.qd_ant) / self.dt / 200.0) ** 2),
            "limite": perto_limite,
            "escorregar": escorrega,
        }
        total = sum(w[k] * val for k, val in partes.items()) * self.dt / 0.02
        return total, partes

    # ------------------------------------------------------------------ render
    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, 480, 640)
                self._camera = mujoco.MjvCamera()
                self._camera.trackbodyid = self.torso
                self._camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self._camera.distance, self._camera.elevation, self._camera.azimuth = 0.7, -15, 120
            self._renderer.update_scene(self.data, self._camera)
            return self._renderer.render()
        if self.render_mode == "human":
            if self._viewer is None:
                from mujoco import viewer
                self._viewer = viewer.launch_passive(self.model, self.data)
                self._viewer.cam.trackbodyid = self.torso
                self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self._viewer.cam.distance = 0.7
            self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
