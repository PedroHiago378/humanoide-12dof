"""Politica treinada (PPO) rodando so com numpy, a partir do .npz de rl/exportar.py."""
import numpy as np


class PoliticaNumpy:
    def __init__(self, caminho):
        with np.load(caminho) as d:   # fecha o arquivo: o treino precisa sobrescreve-lo
            self.camadas = [(d[f"W{i}"], d[f"b{i}"]) for i in range(int(d["n_camadas"]))]
            self.media, self.var = d["obs_media"], d["obs_var"]
            self.clip, self.eps = float(d["clip_obs"]), float(d["epsilon"])
            self.passos = int(d["passos"]) if "passos" in d else 0
        self.caminho = caminho

    def __call__(self, obs):
        x = np.clip((np.asarray(obs, np.float32) - self.media) / np.sqrt(self.var + self.eps), -self.clip, self.clip)
        for W, b in self.camadas[:-1]:
            x = np.tanh(W @ x + b)
        W, b = self.camadas[-1]
        return np.clip(W @ x + b, -1.0, 1.0)
