"""Dinamica de corpo rigido do humanoide com base (torso) fixa.

- rnea:   dinamica inversa (Newton-Euler) -> torques das juntas
- massa:  matriz de massa (algoritmo do corpo rigido composto, CRBA)
- direta: dinamica direta qdd = M^-1 (tau - h)

Tudo no frame da base. As recursoes ao longo da arvore viram produtos pela matriz
robo.caminho (caminho[i, k] = 1 se a junta k esta entre a base e o elo i), o que
deixa tudo vetorizado em numpy.

Gravidade padrao = robo de pe (torso vertical); passe gravidade=R_base.T @ [0, 0, -9.81]
se o torso estiver inclinado.
"""
import numpy as np

G = np.array([0.0, 0.0, -9.81])


def _no_pai(robo, x):
    """x[pai(i)] para cada junta i; zero quando o pai e a base."""
    return np.where((robo.pai >= 0)[:, None], x[robo.pai], 0.0)


def rnea(robo, q, qd, qdd, gravidade=G, cinematica=None):
    """Torques tau = M(q) qdd + C(q, qd) qd + g(q)."""
    R, p, a = cinematica or robo.fk(q)
    C = robo.caminho
    # cinematica: velocidade/aceleracao angular de cada elo e aceleracao linear da sua origem
    w = C @ (a * qd[:, None])
    wp = _no_pai(robo, w)
    dw = C @ (a * qdd[:, None] + np.cross(wp, a * qd[:, None]))
    dwp = _no_pai(robo, dw)
    r = p - _no_pai(robo, p)
    acc = -gravidade + C @ (np.cross(dwp, r) + np.cross(wp, np.cross(wp, r)))

    # forcas/momentos de cada elo e soma sobre as subarvores
    c = np.einsum("nij,nj->ni", R, robo.com)                 # COM relativo a origem do elo
    Iw = R @ robo.I @ R.transpose(0, 2, 1)
    ac = acc + np.cross(dw, c) + np.cross(w, np.cross(w, c))
    F = robo.m[:, None] * ac
    N = np.einsum("nij,nj->ni", Iw, dw) + np.cross(w, np.einsum("nij,nj->ni", Iw, w))
    f = C.T @ F
    m = C.T @ (N + np.cross(p + c, F)) - np.cross(p, f)       # momento em torno de cada junta
    return np.einsum("ni,ni->n", a, m)


def gravidade(robo, q, gravidade=G):
    z = np.zeros(robo.n)
    return rnea(robo, q, z, z, gravidade)


def massa(robo, q, cinematica=None):
    """Matriz de massa via CRBA em coordenadas espaciais da base ([angular; linear])."""
    R, p, a = cinematica or robo.fk(q)
    n = robo.n
    S = np.hstack([a, np.cross(p, a)])
    c = p + np.einsum("nij,nj->ni", R, robo.com)
    Cx = np.zeros((n, 3, 3))
    Cx[:, 0, 1], Cx[:, 0, 2], Cx[:, 1, 2] = -c[:, 2], c[:, 1], -c[:, 0]
    Cx[:, 1, 0], Cx[:, 2, 0], Cx[:, 2, 1] = c[:, 2], -c[:, 1], c[:, 0]
    m = robo.m[:, None, None]
    I = np.zeros((n, 6, 6))
    I[:, :3, :3] = R @ robo.I @ R.transpose(0, 2, 1) + m * Cx @ Cx.transpose(0, 2, 1)
    I[:, :3, 3:] = m * Cx
    I[:, 3:, :3] = m * Cx.transpose(0, 2, 1)
    I[:, 3:, 3:] = m * np.eye(3)
    Ic = np.einsum("ki,kab->iab", robo.caminho, I)            # inercia composta da subarvore
    P = S @ np.einsum("jab,jb->ja", Ic, S).T                  # P[k, j] = S_k . Ic_j S_j
    U = P * robo.caminho.T                                     # vale quando k e ancestral de j
    return U + U.T - np.diag(np.diag(P))


def direta(robo, q, qd, tau, gravidade=G, armadura=None):
    """Aceleracao das juntas. armadura = inercia refletida do motor (kg.m2) por junta."""
    cin = robo.fk(q)
    M = massa(robo, q, cin)
    if armadura is not None:
        M = M + np.diag(np.broadcast_to(armadura, robo.n))
    h = rnea(robo, q, qd, np.zeros(robo.n), gravidade, cin)
    return np.linalg.solve(M, tau - h)
