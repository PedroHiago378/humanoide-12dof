"""Exporta uma politica PPO (SB3) para .npz, rodavel so com numpy (controle/politica_rl.py).

    python rl/exportar.py rl/execucoes/<nome>/modelo_final.zip            # -> politica.npz na mesma pasta
    python rl/exportar.py <checkpoint.zip> --saida <arquivo.npz>

Conteudo: media/variancia da normalizacao das observacoes (VecNormalize), camadas da rede
do ator (Linear + tanh) e a camada de saida (media da acao).
"""
import argparse
import os

import numpy as np


def exportar(modelo_zip, vecnormalize=None, saida=None):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    pasta = os.path.dirname(os.path.abspath(modelo_zip))
    vecnormalize = vecnormalize or os.path.join(pasta, "vecnormalize.pkl")
    saida = saida or os.path.join(pasta, "politica.npz")
    modelo = PPO.load(modelo_zip, device="cpu")
    pol = modelo.policy
    camadas = [m for m in pol.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)]
    ativacoes = {type(m).__name__ for m in pol.mlp_extractor.policy_net if not isinstance(m, torch.nn.Linear)}
    if ativacoes - {"Tanh"}:
        raise ValueError(f"ativacao nao suportada: {ativacoes}")
    dados = {"n_camadas": len(camadas) + 1}
    for i, c in enumerate(camadas + [pol.action_net]):
        dados[f"W{i}"] = c.weight.detach().numpy().astype(np.float32)
        dados[f"b{i}"] = c.bias.detach().numpy().astype(np.float32)
    if os.path.exists(vecnormalize):
        with open(vecnormalize, "rb") as fp:
            import pickle
            norm = pickle.load(fp)
        dados.update(obs_media=norm.obs_rms.mean.astype(np.float32), obs_var=norm.obs_rms.var.astype(np.float32),
                     clip_obs=np.float32(norm.clip_obs), epsilon=np.float32(norm.epsilon))
    else:
        n = camadas[0].in_features
        dados.update(obs_media=np.zeros(n, np.float32), obs_var=np.ones(n, np.float32),
                     clip_obs=np.float32(np.inf), epsilon=np.float32(0.0))
    dados["passos"] = np.int64(modelo.num_timesteps)
    tmp = saida + ".tmp.npz"
    np.savez(tmp, **dados)
    import time
    for tentativa in range(20):   # no Windows falha se o arquivo estiver aberto por outro processo
        try:
            os.replace(tmp, saida)
            break
        except PermissionError:
            if tentativa == 19:
                raise
            time.sleep(0.1)
    return saida


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modelo")
    ap.add_argument("--vecnormalize")
    ap.add_argument("--saida")
    args = ap.parse_args()
    print("Exportado:", exportar(args.modelo, args.vecnormalize, args.saida))


if __name__ == "__main__":
    main()
