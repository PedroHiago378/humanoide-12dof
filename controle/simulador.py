"""Simulacao dinamica com base fixa e modelo dos servos MG90S.

Os servos de hobby sao controlados em posicao: recebem um angulo e um laco interno
(aqui um PD com saturacao) gera o torque. O simulador aceita os dois tipos de comando:
  passo(q_ref=...)  -> servo interno gera o torque (igual ao hardware real)
  passo(tau=...)    -> torque direto (para testar controladores de torque)
"""
from dataclasses import dataclass, field

import numpy as np

from . import dinamica


@dataclass
class ModeloServo:
    kp: float = 2.4            # N.m/rad  (satura com ~5 graus de erro)
    kd: float = 0.05           # N.m.s/rad
    armadura: float = 1e-3     # kg.m2 inercia do motor refletida na saida (estimativa)
    atrito_viscoso: float = 0.01

    def torque(self, robo, q_ref, q, qd):
        tau = self.kp * (q_ref - q) - self.kd * qd
        # curva linear torque x velocidade do motor DC
        sup = robo.torque_max * np.clip(1 - qd / robo.velocidade_max, 0, 2)
        inf = -robo.torque_max * np.clip(1 + qd / robo.velocidade_max, 0, 2)
        return np.clip(tau, inf, sup)


@dataclass
class Estado:
    t: float
    q: np.ndarray
    qd: np.ndarray
    tau: np.ndarray = field(default=None)


class SimuladorBaseFixa:
    def __init__(self, robo, dt=0.002, servo=None, gravidade=dinamica.G):
        self.robo = robo
        self.dt = dt
        self.servo = servo or ModeloServo()
        self.gravidade = np.asarray(gravidade, float)
        self.reiniciar()

    def reiniciar(self, q=None):
        n = self.robo.n
        self.estado = Estado(0.0, np.zeros(n) if q is None else np.array(q, float), np.zeros(n), np.zeros(n))

    def passo(self, q_ref=None, tau=None):
        r, e, s = self.robo, self.estado, self.servo
        if tau is None:
            tau = s.torque(r, r.limitar(q_ref), e.q, e.qd)
        tau = np.clip(tau, -r.torque_max, r.torque_max)
        qdd = dinamica.direta(r, e.q, e.qd, tau - s.atrito_viscoso * e.qd, self.gravidade, s.armadura)
        qd = e.qd + qdd * self.dt           # Euler semi-implicito
        q = e.q + qd * self.dt
        no_limite = (q < r.limite[:, 0]) | (q > r.limite[:, 1])
        qd[no_limite] = 0.0
        self.estado = Estado(e.t + self.dt, r.limitar(q), qd, tau)
        return self.estado
