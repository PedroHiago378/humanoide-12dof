"""Controle de equilibrio pela IMU (MPU-6050 na ESP32).

Fluxo, a cada quadro do servidor (50 Hz):
  amostras brutas (IMUSTREAM) -> frame do corpo (montagem do sensor, calibrada em "nivelar")
  -> filtro complementar -> inclinacao pitch/roll -> PID por eixo -> correcao somada as juntas.

Convencao (frame do robo: X frente, Y esquerda, Z cima):
  pitch > 0 = robo caindo para a FRENTE (giro em +Y)
  roll  > 0 = robo caindo para a DIREITA (giro em +X)
A correcao de cada junta e  peso * u, com u = Kp*e + Ki*int(e) + Kd*de/dt (graus).
Com pe no chao, peso +1 no tornozelo/quadril pitch empurra contra a queda para frente
(o mesmo vale para quadril_roll no roll). Se a junta empurrar a favor da queda, inverta o peso.
"""
import copy
import json
import os
import time

import numpy as np

from .modelo import DADOS

ARQUIVO = os.path.join(DADOS, "equilibrio.json")
EIXOS_SENSOR = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0),
                "+z": (0, 0, 1), "-z": (0, 0, -1)}
VALIDADE_IMU = 0.25   # s sem amostra -> sem IMU, correcao volta a zero

PADRAO = {
    "ativo": False,
    "sensor": {
        "frente": "+x",       # eixo do sensor que aponta para a frente do robo
        "alfa": 0.98,         # filtro complementar: peso do giroscopio
        "inverter_pitch": False,
        "inverter_roll": False,
        "R": None,            # sensor -> corpo (3x3), calculada no nivelar
        "bias_giro": [0.0, 0.0, 0.0],
    },
    "eixos": {
        "pitch": {"ativo": True, "kp": 0.8, "ki": 0.0, "kd": 0.03, "limite": 20.0, "zona_morta": 0.5,
                  "juntas": {"tornozelo_L": 1.0, "tornozelo_R": 1.0}},
        "roll": {"ativo": False, "kp": 0.6, "ki": 0.0, "kd": 0.02, "limite": 12.0, "zona_morta": 0.5,
                 "juntas": {"quadril_roll_L": 1.0, "quadril_roll_R": 1.0}},
    },
    "queda": 45.0,        # graus: acima disso desiste (correcao volta a zero)
    "taxa_max": 300.0,    # graus/s: limite de variacao da correcao
}


def _mesclar(base, novo):
    for k, v in novo.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict) and k != "juntas":
            _mesclar(base[k], v)
        else:
            base[k] = v
    return base


class Equilibrio:
    def __init__(self, robo):
        self.robo = robo
        self.cfg = copy.deepcopy(PADRAO)
        try:
            with open(ARQUIVO, encoding="utf-8") as fp:
                _mesclar(self.cfg, json.load(fp))
        except (OSError, ValueError):
            pass
        self.angulo = np.zeros(2)          # pitch, roll (graus)
        self.taxa = np.zeros(2)            # graus/s
        self.t_ultima = None               # t_ms da ultima amostra (relogio da ESP32)
        self.t_recebida = 0.0              # time.monotonic() da ultima amostra
        self.simulado = None               # (pitch, roll) para testar sem robo
        self.integral = np.zeros(2)
        self.u = np.zeros(2)
        self.corr = np.zeros(robo.n)       # rad, ja limitada pela taxa
        self.queda = False

    # ---------------------------------------------------------------- configuracao
    def configurar(self, parcial):
        _mesclar(self.cfg, parcial)
        for eixo in self.cfg["eixos"].values():
            eixo["juntas"] = {j: float(p) for j, p in eixo["juntas"].items() if j in self.robo.indice}
        self.salvar()

    def salvar(self):
        os.makedirs(DADOS, exist_ok=True)
        with open(ARQUIVO, "w", encoding="utf-8") as fp:
            json.dump(self.cfg, fp, indent=1)

    def nivelar(self, amostras):
        """Robo em pe e parado: a media da aceleracao define o "cima" e a do giro o bias."""
        if len(amostras) < 10:
            raise RuntimeError("poucas amostras da IMU (ela esta transmitindo?)")
        acel = np.mean([a for _, a, _ in amostras], axis=0)
        giro = np.mean([g for _, _, g in amostras], axis=0)
        cima = acel / np.linalg.norm(acel)
        f = np.array(EIXOS_SENSOR[self.cfg["sensor"]["frente"]], float)
        f = f - (f @ cima) * cima
        if np.linalg.norm(f) < 0.3:
            raise RuntimeError("o eixo 'frente' do sensor esta quase vertical; escolha outro")
        f /= np.linalg.norm(f)
        esq = np.cross(cima, f)
        self.cfg["sensor"]["R"] = np.array([f, esq, cima]).round(6).tolist()
        self.cfg["sensor"]["bias_giro"] = giro.round(4).tolist()
        self.angulo[:] = 0
        self.integral[:] = 0
        self.salvar()

    # ---------------------------------------------------------------- estimativa
    def amostra(self, t_ms, acel, giro):
        s = self.cfg["sensor"]
        if s["R"] is None:
            return
        R = np.array(s["R"])
        a = R @ acel                                   # (frente, esquerda, cima) no corpo
        w = R @ (giro - np.array(s["bias_giro"]))      # graus/s no corpo
        sinais = np.array([-1.0 if s["inverter_pitch"] else 1.0, -1.0 if s["inverter_roll"] else 1.0])
        acc = np.degrees([np.arctan2(-a[0], a[2]), np.arctan2(a[1], a[2])]) * sinais
        taxa = np.array([w[1], w[0]]) * sinais
        dt = 0.0 if self.t_ultima is None else (t_ms - self.t_ultima) / 1000
        if not 0 < dt < 0.2:   # primeira amostra ou buraco no fluxo: recomeca pelo acelerometro
            self.angulo = acc
        else:
            alfa = s["alfa"]
            self.angulo = alfa * (self.angulo + taxa * dt) + (1 - alfa) * acc
        self.taxa = taxa
        self.t_ultima = t_ms
        self.t_recebida = time.monotonic()

    def fonte(self):
        if time.monotonic() - self.t_recebida < VALIDADE_IMU and self.cfg["sensor"]["R"] is not None:
            return "imu"
        return "simulado" if self.simulado is not None else None

    def medir(self, dt):
        """Inclinacao e taxa atuais (graus, graus/s) ou None."""
        fonte = self.fonte()
        if fonte == "imu":
            return self.angulo.copy(), self.taxa.copy()
        if fonte == "simulado":
            novo = np.array(self.simulado, float)
            taxa = (novo - self.angulo) / dt if dt > 0 else np.zeros(2)
            self.angulo = novo
            return novo, taxa
        return None

    # ---------------------------------------------------------------- controle
    def correcao(self, dt):
        """Correcao (rad) a somar no comando de posicao das juntas."""
        alvo = np.zeros(2)
        medida = self.medir(dt)
        self.queda = medida is not None and np.abs(medida[0]).max() > self.cfg["queda"]
        if self.cfg["ativo"] and medida is not None and not self.queda:
            ang, taxa = medida
            for i, nome in enumerate(("pitch", "roll")):
                e = self.cfg["eixos"][nome]
                if not e["ativo"]:
                    self.integral[i] = 0
                    continue
                erro = ang[i] if abs(ang[i]) > e["zona_morta"] else 0.0
                if e["ki"] > 0:
                    self.integral[i] = np.clip(self.integral[i] + erro * dt, -e["limite"] / e["ki"], e["limite"] / e["ki"])
                else:
                    self.integral[i] = 0
                alvo[i] = np.clip(e["kp"] * erro + e["ki"] * self.integral[i] + e["kd"] * taxa[i],
                                  -e["limite"], e["limite"])
        else:
            self.integral[:] = 0
        self.u = alvo
        desejada = np.zeros(self.robo.n)
        for i, nome in enumerate(("pitch", "roll")):
            for junta, peso in self.cfg["eixos"][nome]["juntas"].items():
                desejada[self.robo.indice[junta]] += np.radians(peso * alvo[i])
        passo = np.radians(self.cfg["taxa_max"]) * dt
        self.corr += np.clip(desejada - self.corr, -passo, passo)
        return self.corr

    def estado_ui(self):
        fonte = self.fonte()
        return {
            "ativo": self.cfg["ativo"], "fonte": fonte, "queda": bool(self.queda),
            "nivelado": self.cfg["sensor"]["R"] is not None,
            "pitch": round(float(self.angulo[0]), 2) if fonte else None,
            "roll": round(float(self.angulo[1]), 2) if fonte else None,
            "u": np.round(self.u, 2).tolist(),
            "correcao": {n: round(float(np.degrees(c)), 1) for n, c in zip(self.robo.nomes, self.corr) if abs(c) > 1e-4},
        }
