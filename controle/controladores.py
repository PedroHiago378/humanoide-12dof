"""Controladores. Para criar um novo: herde de Controlador, implemente calcular()
e registre em CONTROLADORES.

tipo = "posicao": calcular() devolve q_ref (rad) -> vai para os servos reais e para o simulador.
tipo = "torque":  calcular() devolve tau (N.m)   -> so no simulador (servo de hobby nao aceita torque).
"""
import numpy as np

from . import animacao, biblioteca, dinamica


def pose_da_msg(msg):
    """Pose de uma mensagem {"q": {junta: graus}} ou {"nome": pose da biblioteca}."""
    return msg["q"] if "q" in msg else biblioteca.pose(msg["nome"])


class Controlador:
    nome = "base"
    tipo = "posicao"

    def __init__(self, robo):
        self.robo = robo

    def reiniciar(self, estado):
        """Chamado ao ativar o controlador."""

    def comando(self, msg):
        """Mensagens do visualizador ({"tipo": "alvo"|"pose", ...})."""

    def calcular(self, t, estado):
        raise NotImplementedError


class Manual(Controlador):
    """Segue alvos dos sliders e poses. Pose aceita "duracao" (s) e "interp" (animacao.TIPOS)."""
    nome = "manual"
    duracao_slider = 0.3

    def reiniciar(self, estado):
        self.ini = estado.q.copy()
        self.alvo = estado.q.copy()
        self.t0 = estado.t
        self.t = estado.t
        self.duracao, self.interp = self.duracao_slider, "min_jerk"

    def _novo_alvo(self, alvo, duracao, interp):
        self.ini = self._ref(self.t)
        self.alvo = self.robo.limitar(alvo)
        self.t0 = self.t
        self.duracao, self.interp = max(float(duracao), 1e-3), interp

    def comando(self, msg):
        if msg["tipo"] == "alvo":
            alvo = self.alvo.copy()
            for nome, graus in msg["q"].items():
                alvo[self.robo.indice[nome]] = np.radians(graus)
            self._novo_alvo(alvo, self.duracao_slider, "min_jerk")
        elif msg["tipo"] == "pose":
            self._novo_alvo(self.robo.q_de(pose_da_msg(msg)), msg.get("duracao", 0.8), msg.get("interp", "min_jerk"))

    def _ref(self, t):
        s = (t - self.t0) / self.duracao
        tipo = "suave" if self.interp == "spline" else self.interp
        return self.ini + (self.alvo - self.ini) * animacao.forma(tipo, s)

    def calcular(self, t, estado):
        self.t = t
        return self._ref(t)


class Agachamento(Controlador):
    """Agacha e levanta periodicamente mantendo o pe paralelo ao chao."""
    nome = "agachamento"
    amplitude = np.radians(35)
    periodo = 3.0

    def calcular(self, t, estado):
        r = self.robo
        a = self.amplitude * 0.5 * (1 - np.cos(2 * np.pi * t / self.periodo))
        q = r.q_de({"cotovelo_L": -80, "cotovelo_R": 80})
        for lado in "LR":
            q[r.indice[f"quadril_pitch_{lado}"]] = -a
            q[r.indice[f"joelho_{lado}"]] = 2 * a
            q[r.indice[f"tornozelo_{lado}"]] = -a
        return q


class Acenar(Controlador):
    nome = "acenar"

    def calcular(self, t, estado):
        r = self.robo
        q = r.q_de({"cotovelo_R": 80, "ombro_L": 0})
        q[r.indice["cotovelo_L"]] = np.radians(60 + 25 * np.sin(2 * np.pi * 1.2 * t))
        return q


class PDCompensacaoGravidade(Controlador):
    """Exemplo de controle por torque: tau = Kp (q* - q) - Kd qd + g(q). So no simulador."""
    nome = "pd_gravidade (torque)"
    tipo = "torque"
    kp, kd = 1.5, 0.04

    def reiniciar(self, estado):
        self.alvo = self.robo.q_de(biblioteca.poses_fabrica()["agachar"])

    def comando(self, msg):
        if msg["tipo"] == "pose":
            self.alvo = self.robo.q_de(pose_da_msg(msg))

    def calcular(self, t, estado):
        g = dinamica.gravidade(self.robo, estado.q)
        return self.kp * (self.alvo - estado.q) - self.kd * estado.qd + g


class Animacao(Controlador):
    """Toca uma animacao por keyframe (animacao.py). Entra e salta (scrub) com transicao suave
    para nao dar tranco nos servos.
    Mensagens: anim_tocar {anim, inicio, loop} | anim_parar | anim_ir {anim, t}."""
    nome = "animacao"
    entrada = 0.8      # s de transicao ao comecar a tocar
    salto = 0.25       # s de transicao ao arrastar o cursor

    def reiniciar(self, estado):
        self.anim = None
        self.tocando = False
        self.loop = False
        self.t_anim = 0.0
        self.t = estado.t
        self.q_fixo = estado.q.copy()
        self._transicao(estado.q, 0.0)

    def _transicao(self, de, dur):
        self.trans_de, self.trans_t0, self.trans_dur = np.array(de, float), self.t, dur

    def comando(self, msg):
        tipo = msg["tipo"]
        atual = self._saida(self.t)
        if tipo == "anim_tocar":
            self.anim, self.loop = msg["anim"], bool(msg.get("loop"))
            self.t_anim = float(msg.get("inicio", 0.0))
            self.tocando = True
            self.t_inicio = self.t + self.entrada - self.t_anim   # segura o inicio durante a entrada
            self._transicao(atual, self.entrada)
        elif tipo == "anim_parar":
            self.tocando = False
            self.q_fixo, self.anim = atual, None
            self._transicao(atual, 0.0)
        elif tipo == "anim_ir":
            self.anim, self.tocando = msg["anim"], False
            self.t_anim = float(msg["t"])
            self._transicao(atual, self.salto)

    def _alvo(self):
        if self.anim is None:
            return self.q_fixo
        return np.radians(animacao.avaliar(self.anim, self.t_anim, self.robo.nomes))

    def _saida(self, t):
        s = (t - self.trans_t0) / self.trans_dur if self.trans_dur > 0 else 1.0
        return self.trans_de + (self._alvo() - self.trans_de) * animacao.forma("min_jerk", s)

    def calcular(self, t, estado):
        self.t = t
        if self.tocando and self.anim:
            dur = animacao.duracao(self.anim)
            ta = max(t - self.t_inicio, self.t_anim if t < self.t_inicio else 0.0)
            if self.loop and dur > 0:
                ta %= dur
            elif ta >= dur:
                ta, self.tocando = dur, False
            self.t_anim = ta
        return self.robo.limitar(self._saida(t))

    def estado_ui(self):
        return {"t": self.t_anim, "tocando": self.tocando, "loop": self.loop}


CONTROLADORES = [Manual, Animacao, Agachamento, Acenar, PDCompensacaoGravidade]
