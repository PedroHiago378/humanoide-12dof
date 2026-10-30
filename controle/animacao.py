"""Animacoes por keyframe e formas de interpolacao.

Formato (o mesmo salvo na biblioteca e usado pelo visualizador):
    {
      "interp": "suave",                     # padrao da animacao
      "interp_juntas": {"joelho_L": "spline"},   # sobrepoe o keyframe para essa junta
      "keyframes": [
        {"t": 0.0, "q": {"joelho_L": 0, ...}},
        {"t": 1.2, "q": {...}, "interp": "linear"}   # forma do trecho que CHEGA neste keyframe
      ]
    }
Angulos em graus de junta, tempo em segundos.
Mantenha em sincronia com visualizador/animacao.js (avaliar / forma).
"""
import numpy as np

TIPOS = {
    "linear": "linear",
    "suave": "suave (cosseno)",
    "min_jerk": "jerk minimo",
    "acelerar": "acelerar",
    "desacelerar": "desacelerar",
    "degrau": "degrau (segura e pula)",
    "spline": "spline (passa suave pelos keyframes)",
}


def forma(tipo, s):
    """Progresso 0..1 do trecho para cada tipo (exceto spline, tratada em avaliar)."""
    s = np.clip(s, 0.0, 1.0)
    if tipo == "linear":
        return s
    if tipo == "min_jerk":
        return s**3 * (10 - 15 * s + 6 * s**2)
    if tipo == "acelerar":
        return s * s
    if tipo == "desacelerar":
        return 1 - (1 - s) ** 2
    if tipo == "degrau":
        return np.where(s >= 1.0, 1.0, 0.0)
    return (1 - np.cos(np.pi * s)) / 2  # suave


def duracao(anim):
    ks = anim.get("keyframes") or []
    return max((k["t"] for k in ks), default=0.0)


def avaliar(anim, t, nomes):
    """Pose (graus) em t para as juntas `nomes`."""
    ks = sorted(anim.get("keyframes") or [], key=lambda k: k["t"])
    if not ks:
        return np.zeros(len(nomes))
    T = np.array([k["t"] for k in ks], float)
    P = np.array([[k["q"].get(n, 0.0) for n in nomes] for k in ks], float)  # (nk, nj)
    if len(ks) == 1 or t <= T[0]:
        return P[0].copy()
    if t >= T[-1]:
        return P[-1].copy()
    i = int(np.searchsorted(T, t, side="right"))   # trecho ks[i-1] -> ks[i]
    t0, t1 = T[i - 1], T[i]
    h = t1 - t0
    s = (t - t0) / h if h > 0 else 1.0
    padrao = ks[i].get("interp") or anim.get("interp") or "suave"
    sobre = anim.get("interp_juntas") or {}
    out = np.empty(len(nomes))
    for j, n in enumerate(nomes):
        tipo = sobre.get(n) or padrao
        p0, p1 = P[i - 1, j], P[i, j]
        if tipo == "spline":
            # Hermite cubica com tangentes de Catmull-Rom (tempo nao uniforme); pontas com velocidade zero
            m0 = (P[i, j] - P[i - 2, j]) / (T[i] - T[i - 2]) if i >= 2 and T[i] > T[i - 2] else 0.0
            m1 = (P[i + 1, j] - P[i - 1, j]) / (T[i + 1] - T[i - 1]) if i + 1 < len(ks) and T[i + 1] > T[i - 1] else 0.0
            s2, s3 = s * s, s * s * s
            out[j] = ((2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * h * m0
                      + (-2 * s3 + 3 * s2) * p1 + (s3 - s2) * h * m1)
        else:
            out[j] = p0 + (p1 - p0) * forma(tipo, s)
    return out
