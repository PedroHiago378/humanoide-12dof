"""Modelo cinematico do humanoide carregado de modelo/robo.json.

Todos os vetores ficam no frame da base (torso): X frente, Y esquerda, Z cima.
Na pose zero todos os frames dos elos tem a mesma orientacao do torso.
"""
import json
import os
import sys

import numpy as np

if getattr(sys, "frozen", False):  # executavel do PyInstaller: arquivos extraidos em _MEIPASS
    RAIZ = sys._MEIPASS
    DADOS = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "Humanoide")
else:
    RAIZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    DADOS = os.path.join(RAIZ, "config")


def rot(eixo, ang):
    """Rotacao de Rodrigues; eixo (..., 3) unitario e ang (...) podem vir em lote."""
    eixo = np.asarray(eixo, float)
    ang = np.asarray(ang, float)[..., None, None]
    K = np.zeros(eixo.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -eixo[..., 2], eixo[..., 1], -eixo[..., 0]
    K[..., 1, 0], K[..., 2, 0], K[..., 2, 1] = eixo[..., 2], -eixo[..., 1], eixo[..., 0]
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


class Robo:
    def __init__(self, caminho=os.path.join(RAIZ, "modelo", "robo.json")):
        with open(caminho, encoding="utf-8") as fp:
            self.dados = json.load(fp)
        d = self.dados
        js = d["juntas"]
        self.n = len(js)
        self.nomes = [j["nome"] for j in js]
        self.indice = {n: i for i, n in enumerate(self.nomes)}
        self.base = d["base"]
        self.massa_total = d["massa_total"]

        # elo filho de cada junta; pai = indice da junta que move o elo pai (-1 = base)
        self.elo = [j["filho"] for j in js]
        junta_do_elo = {e: i for i, e in enumerate(self.elo)}
        self.pai = np.array([junta_do_elo.get(j["pai"], -1) for j in js])
        assert all(self.pai[i] < i for i in range(self.n)), "juntas devem vir com pai antes do filho"
        self.origem = np.array([j["origem"] for j in js], float)
        self.eixo = np.array([j["eixo"] for j in js], float)
        self.limite = np.array([j["limite"] for j in js], float)
        self.torque_max = np.array([j["torque_max"] for j in js], float)
        self.velocidade_max = np.array([j["velocidade_max"] for j in js], float)
        self.sentido_servo = np.array([j["sentido_servo"] for j in js], int)

        def inercial(nome):
            e = d["elos"][nome]
            return e["massa"], np.array(e["com"], float), np.array(e["inercia"], float)

        self.m_base, self.com_base, self.I_base = inercial(self.base)
        props = [inercial(e) for e in self.elo]
        self.m = np.array([p[0] for p in props])
        self.com = np.array([p[1] for p in props])
        self.I = np.array([p[2] for p in props])
        # subarvore[i] = juntas cujo elo e movido pela junta i (inclui i)
        self.subarvore = [[k for k in range(self.n) if self._ancestral(i, k)] for i in range(self.n)]
        # caminho[i, k] = 1 se a junta k esta no caminho base -> elo i (k ancestral de i ou k == i)
        self.caminho = np.array([[float(self._ancestral(k, i)) for k in range(self.n)] for i in range(self.n)])

    def _ancestral(self, i, k):
        while k >= 0:
            if k == i:
                return True
            k = self.pai[k]
        return False

    def q_de(self, graus: dict):
        """dict {junta: graus} -> vetor q (rad), juntas omitidas = 0."""
        q = np.zeros(self.n)
        for nome, v in graus.items():
            if nome in self.indice:
                q[self.indice[nome]] = np.radians(v)
        return q

    def dict_de(self, q, graus=True):
        return {n: float(np.degrees(v) if graus else v) for n, v in zip(self.nomes, q)}

    def limitar(self, q):
        return np.clip(q, self.limite[:, 0], self.limite[:, 1])

    def fk(self, q):
        """Retorna (R, p, a): orientacao (n,3,3) e posicao (n,3) do frame de cada elo filho,
        e eixo de cada junta (n,3), tudo no frame da base."""
        Rq = rot(self.eixo, q)
        R = np.empty((self.n, 3, 3))
        p = np.empty((self.n, 3))
        a = np.empty((self.n, 3))
        for i in range(self.n):
            k = self.pai[i]
            if k < 0:
                a[i], p[i], R[i] = self.eixo[i], self.origem[i], Rq[i]
            else:
                a[i] = R[k] @ self.eixo[i]
                p[i] = p[k] + R[k] @ self.origem[i]
                R[i] = R[k] @ Rq[i]
        return R, p, a

    def pose_elo(self, q, nome):
        if nome == self.base:
            return np.eye(3), np.zeros(3)
        R, p, _ = self.fk(q)
        i = self.elo.index(nome)
        return R[i], p[i]

    def centro_de_massa(self, q):
        R, p, _ = self.fk(q)
        c = self.m_base * self.com_base + sum(self.m[i] * (p[i] + R[i] @ self.com[i]) for i in range(self.n))
        return c / self.massa_total

    def jacobiano(self, q, nome_elo, ponto_local=np.zeros(3)):
        """Jacobiano (6 x n) [linear; angular] de um ponto do elo, no frame da base."""
        R, p, a = self.fk(q)
        J = np.zeros((6, self.n))
        if nome_elo == self.base:
            return J
        e = self.elo.index(nome_elo)
        alvo = p[e] + R[e] @ ponto_local
        for i in range(self.n):
            if e in self.subarvore[i]:
                J[:3, i] = np.cross(a[i], alvo - p[i])
                J[3:, i] = a[i]
        return J

    def jacobiano_com(self, q):
        R, p, a = self.fk(q)
        J = np.zeros((3, self.n))
        for i in range(self.n):
            for k in self.subarvore[i]:
                J[:, i] += self.m[k] * np.cross(a[i], p[k] + R[k] @ self.com[k] - p[i])
        return J / self.massa_total
