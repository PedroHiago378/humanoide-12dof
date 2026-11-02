"""Treina a caminhada com PPO (Stable-Baselines3) em varios ambientes em paralelo.

    python rl/treinar.py                          # treino completo (padrao 20 M passos, ~2-3 h com 10 ambientes)
    python rl/treinar.py --passos 200000 --envs 4 # teste rapido
    python rl/treinar.py --continuar rl/execucoes/<nome>/modelo_final.zip --passos 5000000
    python rl/treinar.py --config minha_config.json   # sobrepoe PADRAO de ambiente.py
    python rl/treinar.py --config rl/configs/v4_omni.json --iniciar-de <v3>/modelo_final.zip
        # comeca com os pesos de outro modelo mesmo se a observacao cresceu (45 -> 47, modo omni)

Saida em <pasta>/<nome>/ (padrao rl/execucoes): checkpoints, modelo_final.zip, vecnormalize.pkl,
config.json, politica.npz (para o programa, so numpy) e progresso.json (lido pela aba de treino).
Para parar salvando: crie o arquivo PARAR na pasta da execucao (o programa faz isso) ou Ctrl+C.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict, deque

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ambiente import HumanoideAndar  # noqa: E402
from exportar import exportar  # noqa: E402

EXECUCOES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "execucoes")


class Acompanhamento(BaseCallback):
    """Medias das recompensas/velocidade (TensorBoard e progresso.json), checkpoints exportados
    para politica.npz e parada pelo arquivo PARAR."""

    def __init__(self, pasta, envs, total, cada_passos):
        super().__init__()
        self.pasta, self.envs, self.total, self.cada = pasta, envs, total, cada_passos
        self.soma = defaultdict(float)
        self.n = 0
        self.historico = deque(maxlen=400)
        self.t0, self.passos0, self.total_abs = time.time(), 0, total
        self.proximo_json = 0
        self.proximo_ckpt = cada_passos

    def _on_training_start(self):
        self.passos0 = self.num_timesteps
        self.total_abs = self.passos0 + self.total
        self.proximo_ckpt = self.num_timesteps + self.cada   # ao continuar, conta a partir de onde parou

    def _on_step(self):
        for info in self.locals["infos"]:
            for k, v in info.get("recompensas", {}).items():
                self.logger.record_mean(f"recompensa/{k}", v)
                self.soma["r_" + k] += v
            if "vx" in info:
                self.logger.record_mean("caminhada/vx", info["vx"])
                self.logger.record_mean("caminhada/erro_vx", abs(info["vx"] - info["vx_cmd"]))
                self.soma["vx"] += info["vx"]
                self.soma["erro_vx"] += abs(info["vx"] - info["vx_cmd"])
                self.n += 1
        if self.num_timesteps >= self.proximo_json:
            self.proximo_json = self.num_timesteps + 20_000
            self._progresso("treinando")
        if self.num_timesteps >= self.proximo_ckpt:
            self.proximo_ckpt += self.cada
            self.salvar()
        return not os.path.exists(os.path.join(self.pasta, "PARAR"))

    def salvar(self):
        """Salva modelo + normalizacao e exporta politica.npz (o programa pode simular durante o treino)."""
        self.model.save(os.path.join(self.pasta, "modelo_atual"))
        self.envs.save(os.path.join(self.pasta, "vecnormalize.pkl"))
        try:
            exportar(os.path.join(self.pasta, "modelo_atual.zip"), saida=os.path.join(self.pasta, "politica.npz"))
        except Exception as e:  # nao derruba o treino por causa da exportacao
            print("falha ao exportar politica:", e)

    def _progresso(self, estado):
        ep = list(self.model.ep_info_buffer)
        media = {k[2:]: v / max(1, self.n) for k, v in self.soma.items() if k.startswith("r_")}
        dados = {
            "estado": estado,
            "passos": int(self.num_timesteps),
            "total": int(self.total_abs),
            "fps": (self.num_timesteps - self.passos0) / max(1e-6, time.time() - self.t0),
            "ep_len": float(np.mean([e["l"] for e in ep])) if ep else None,
            "ep_rew": float(np.mean([e["r"] for e in ep])) if ep else None,
            "vx": self.soma["vx"] / max(1, self.n),
            "erro_vx": self.soma["erro_vx"] / max(1, self.n),
            "recompensas": media,
            "atualizado": time.time(),
        }
        self.historico.append({k: dados[k] for k in ("passos", "ep_len", "ep_rew", "vx")})
        dados["historico"] = list(self.historico)
        self.soma.clear()
        self.n = 0
        tmp = os.path.join(self.pasta, "progresso.json.tmp")
        with open(tmp, "w", encoding="utf-8") as fp:
            json.dump(dados, fp)
        # no Windows a troca falha se alguem (o programa) estiver lendo o arquivo nesse instante:
        # tenta de novo e, se nao der, pula esta atualizacao em vez de derrubar o treino
        for _ in range(10):
            try:
                os.replace(tmp, os.path.join(self.pasta, "progresso.json"))
                return
            except PermissionError:
                time.sleep(0.05)


def transplantar(modelo, envs, zip_antigo):
    """Copia pesos e normalizacao de um modelo com observacao menor (45 -> 47 no modo omni).
    As entradas novas (lado e giro do comando) comecam com peso zero: no inicio a politica age como a antiga."""
    import pickle
    antigo = PPO.load(zip_antigo, device="cpu")
    n_ant, n_novo = antigo.observation_space.shape[0], modelo.observation_space.shape[0]
    if (n_ant, n_novo) == (45, 47):
        mapa = list(range(43)) + [45, 46]     # vx fica em 42; vy e giro entram em 43-44; fase vai para 45-46
    elif n_ant == n_novo:
        mapa = list(range(n_novo))
    else:
        raise ValueError(f"nao sei adaptar observacao {n_ant} -> {n_novo}")
    sd_ant, sd_novo = antigo.policy.state_dict(), modelo.policy.state_dict()
    for k, v in sd_ant.items():
        if sd_novo[k].shape == v.shape:
            sd_novo[k] = v.clone()
        elif v.dim() == 2 and v.shape[1] == n_ant:
            w = torch.zeros_like(sd_novo[k])
            w[:, mapa] = v
            sd_novo[k] = w
        else:
            raise ValueError(f"camada {k} incompativel: {tuple(v.shape)} -> {tuple(sd_novo[k].shape)}")
    modelo.policy.load_state_dict(sd_novo)
    with open(os.path.join(os.path.dirname(zip_antigo), "vecnormalize.pkl"), "rb") as fp:
        norm = pickle.load(fp)
    envs.obs_rms.mean[mapa] = norm.obs_rms.mean
    envs.obs_rms.var[mapa] = norm.obs_rms.var
    envs.obs_rms.count = norm.obs_rms.count
    envs.ret_rms = norm.ret_rms
    print(f"pesos de {zip_antigo} ({n_ant} -> {n_novo} entradas)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passos", type=int, default=20_000_000)
    ap.add_argument("--envs", type=int, default=max(1, min(10, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--config", help="JSON que sobrepoe a configuracao do ambiente (recompensas etc.)")
    ap.add_argument("--continuar", help="modelo .zip para continuar o treino")
    ap.add_argument("--iniciar-de", help="modelo .zip cujos pesos iniciam um treino novo (aceita observacao maior)")
    ap.add_argument("--nome", default=time.strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--pasta", default=EXECUCOES, help="onde criar a pasta da execucao")
    args = ap.parse_args()

    torch.set_num_threads(2)   # rede pequena: mais threads so disputam CPU com os ambientes
    config = {}
    if args.config:
        with open(args.config, encoding="utf-8") as fp:
            config = json.load(fp)
    pasta = os.path.join(args.pasta, args.nome)
    os.makedirs(pasta, exist_ok=True)
    if os.path.exists(os.path.join(pasta, "PARAR")):
        os.remove(os.path.join(pasta, "PARAR"))
    with open(os.path.join(pasta, "config.json"), "w", encoding="utf-8") as fp:
        json.dump(HumanoideAndar(config).cfg, fp, indent=1)

    envs = make_vec_env(HumanoideAndar, n_envs=args.envs, seed=0, env_kwargs={"config": config},
                        vec_env_cls=SubprocVecEnv if args.envs > 1 else None)
    if args.continuar:
        envs = VecNormalize.load(os.path.join(os.path.dirname(args.continuar), "vecnormalize.pkl"), envs)
        modelo = PPO.load(args.continuar, env=envs, tensorboard_log=args.pasta)
    else:
        envs = VecNormalize(envs, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
        modelo = PPO(
            "MlpPolicy", envs, verbose=1, tensorboard_log=args.pasta,
            n_steps=2048 // max(1, args.envs // 4), batch_size=4096, n_epochs=5,
            learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.002, vf_coef=0.5, max_grad_norm=1.0, target_kl=0.03,
            policy_kwargs={"net_arch": {"pi": [256, 128], "vf": [256, 128]}, "log_std_init": -1.0},
            device="cpu",
        )
        if args.iniciar_de:
            transplantar(modelo, envs, args.iniciar_de)

    acomp = Acompanhamento(pasta, envs, args.passos, cada_passos=1_000_000)
    callbacks = [CheckpointCallback(max(1, 2_000_000 // args.envs), pasta, name_prefix="checkpoint"), acomp]
    estado = "erro"
    try:
        modelo.learn(args.passos, callback=callbacks, tb_log_name=args.nome,
                     reset_num_timesteps=not args.continuar, progress_bar=False)
        estado = "parado" if os.path.exists(os.path.join(pasta, "PARAR")) else "concluido"
    except KeyboardInterrupt:
        estado = "parado"
    finally:  # sempre salva o que aprendeu
        modelo.save(os.path.join(pasta, "modelo_final"))
        envs.save(os.path.join(pasta, "vecnormalize.pkl"))
        exportar(os.path.join(pasta, "modelo_final.zip"), saida=os.path.join(pasta, "politica.npz"))
        acomp._progresso(estado)
        envs.close()
        print("Salvo em", pasta)


if __name__ == "__main__":
    main()
