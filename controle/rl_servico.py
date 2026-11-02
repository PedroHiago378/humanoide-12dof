"""Gerencia os treinos de RL (rl/treinar.py) a partir do programa.

O treino precisa de PyTorch + Stable-Baselines3, que nao vao dentro do executavel: ele roda
num processo separado com o Python instalado no PC (escolhido na aba de treino). O progresso
vem do progresso.json que o treinar.py grava; parar = criar o arquivo PARAR (salva antes de sair).
O treino continua mesmo se o programa for fechado; ao abrir de novo ele e reconhecido.
"""
import copy
import glob
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile

from .modelo import DADOS, RAIZ
from .politica_rl import PoliticaNumpy

CONGELADO = getattr(sys, "frozen", False)
BASE_RL = os.path.normpath(os.path.join(DADOS, "rl") if CONGELADO else os.path.join(RAIZ, "rl"))
EXECUCOES = os.path.join(BASE_RL, "execucoes")
ARQ_CONFIG = os.path.join(DADOS, "rl_config.json")
IMPORTS = "import stable_baselines3, torch, mujoco, gymnasium"


def _nome_seguro(nome):
    """Nome de pasta sem caminho nem caracteres estranhos (vem de arquivo recebido)."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(nome or "")).strip("._")[:60]


def _sem_janela():
    return subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


class ServicoTreino:
    def __init__(self):
        if os.path.join(RAIZ, "rl") not in sys.path:
            sys.path.insert(0, os.path.join(RAIZ, "rl"))
        from ambiente import PADRAO
        self.padrao = copy.deepcopy(PADRAO["recompensas"])
        self.config = {"python": "" if CONGELADO else sys.executable, "recompensas": dict(self.padrao),
                       "passos": 20_000_000, "envs": max(1, min(10, (os.cpu_count() or 2) - 2))}
        try:
            with open(ARQ_CONFIG, encoding="utf-8") as fp:
                salvo = json.load(fp)
            self.config.update({k: v for k, v in salvo.items() if k != "recompensas"})
            self.config["recompensas"].update(salvo.get("recompensas", {}))
        except (OSError, ValueError):
            pass
        self.proc = None
        self.ativo = None          # nome da execucao em andamento
        self.ultimo = None         # ultima execucao iniciada (para mostrar o resultado)
        self.python_ok = None      # resultado da ultima verificacao
        os.makedirs(EXECUCOES, exist_ok=True)
        self._reconhecer_ativo()

    # ------------------------------------------------------------------ configuracao
    def salvar_config(self, novo=None):
        if novo:
            if "recompensas" in novo:
                self.config["recompensas"].update({k: float(v) for k, v in novo["recompensas"].items()
                                                   if k in self.padrao})
            for k in ("python", "passos", "envs"):
                if k in novo:
                    self.config[k] = novo[k]
            if "python" in novo:
                self.python_ok = None
        os.makedirs(DADOS, exist_ok=True)
        with open(ARQ_CONFIG, "w", encoding="utf-8") as fp:
            json.dump(self.config, fp, indent=1)

    def restaurar_recompensas(self):
        self.config["recompensas"] = dict(self.padrao)
        self.salvar_config()

    def candidatos_python(self):
        achados = [self.config["python"]] if self.config["python"] else []
        if not CONGELADO:
            achados.append(sys.executable)
        for nome in ("python", "python3", "py"):
            p = shutil.which(nome)
            if p and "WindowsApps" not in p:
                achados.append(p)
        achados += glob.glob(os.path.expandvars(r"%LOCALAPPDATA%\Programs\Python\Python3*\python.exe"))
        vistos = []
        for p in achados:
            if p and p not in vistos:
                vistos.append(p)
        return vistos

    def verificar_python(self):
        """Acha um Python com torch/SB3/MuJoCo (o configurado primeiro)."""
        for p in self.candidatos_python():
            try:
                r = subprocess.run([p, "-c", IMPORTS], capture_output=True, timeout=60, creationflags=_sem_janela())
            except (OSError, subprocess.TimeoutExpired):
                continue
            if r.returncode == 0:
                self.config["python"] = p
                self.salvar_config()
                self.python_ok = p
                return p
        self.python_ok = False
        return None

    # ------------------------------------------------------------------ codigo do treino
    def _codigo(self):
        """Pasta com rl/*.py e modelo/mujoco que o Python externo usa. No executavel os arquivos
        extraidos somem ao fechar, entao copia para os dados do usuario."""
        if not CONGELADO:
            return os.path.join(RAIZ, "rl")
        destino = os.path.join(DADOS, "rl_codigo")
        shutil.copytree(os.path.join(RAIZ, "rl"), os.path.join(destino, "rl"), dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("execucoes", "__pycache__"))
        shutil.copytree(os.path.join(RAIZ, "modelo", "mujoco"), os.path.join(destino, "modelo", "mujoco"),
                        dirs_exist_ok=True)
        return os.path.join(destino, "rl")

    # ------------------------------------------------------------------ treino
    def iniciar(self, nome=None, continuar=None, passos=None, envs=None):
        if self.rodando():
            raise RuntimeError(f"ja ha um treino rodando ({self.ativo})")
        py = self.python_ok or self.verificar_python()
        if not py:
            raise RuntimeError("nenhum Python com torch, stable-baselines3 e mujoco foi encontrado "
                               "(instale: pip install mujoco gymnasium stable-baselines3[extra] torch)")
        nome = (nome or time.strftime("treino_%Y%m%d-%H%M%S")).strip()
        if any(c in nome for c in '\\/:*?"<>|') or not nome:
            raise ValueError("nome de execucao invalido")
        passos = int(passos or self.config["passos"])
        envs = int(envs or self.config["envs"])
        self.salvar_config({"passos": passos, "envs": envs})
        pasta = os.path.join(EXECUCOES, nome)
        os.makedirs(pasta, exist_ok=True)
        arq_cfg = os.path.join(pasta, "config_entrada.json")
        with open(arq_cfg, "w", encoding="utf-8") as fp:
            json.dump({"recompensas": self.config["recompensas"]}, fp, indent=1)
        codigo = self._codigo()
        cmd = [py, os.path.join(codigo, "treinar.py"), "--nome", nome, "--pasta", EXECUCOES,
               "--passos", str(passos), "--envs", str(envs), "--config", arq_cfg]
        if continuar:
            cmd += ["--continuar", self._modelo_da_execucao(continuar)]
        log = open(os.path.join(EXECUCOES, f"{nome}.log"), "w", encoding="utf-8")
        self.proc = subprocess.Popen(cmd, cwd=os.path.dirname(codigo), stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=_sem_janela())
        self.ativo = self.ultimo = nome
        with open(os.path.join(BASE_RL, "ativo.json"), "w", encoding="utf-8") as fp:
            json.dump({"nome": nome, "pid": self.proc.pid, "inicio": time.time()}, fp)
        return nome

    def parar(self):
        if not self.ativo:
            raise RuntimeError("nenhum treino rodando")
        open(os.path.join(EXECUCOES, self.ativo, "PARAR"), "w").close()

    def _reconhecer_ativo(self):
        """Treino iniciado numa sessao anterior do programa e ainda vivo."""
        try:
            with open(os.path.join(BASE_RL, "ativo.json"), encoding="utf-8") as fp:
                nome = json.load(fp)["nome"]
        except (OSError, ValueError, KeyError):
            return
        p = self._progresso(nome)
        if p and p.get("estado") == "treinando" and time.time() - p.get("atualizado", 0) < 120:
            self.ativo = self.ultimo = nome

    def rodando(self):
        if self.proc is not None:
            if self.proc.poll() is None:
                return True
            self.proc = None
            self.ativo = None
            return False
        if self.ativo:
            p = self._progresso(self.ativo)
            if p and p.get("estado") == "treinando" and time.time() - p.get("atualizado", 0) < 120:
                return True
            self.ativo = None
        return False

    def _progresso(self, nome):
        try:
            with open(os.path.join(EXECUCOES, nome, "progresso.json"), encoding="utf-8") as fp:
                return json.load(fp)
        except (OSError, ValueError):
            return None

    def log(self, nome, linhas=12):
        try:
            with open(os.path.join(EXECUCOES, f"{nome}.log"), encoding="utf-8", errors="replace") as fp:
                return [l.rstrip() for l in fp.readlines()[-linhas:]]
        except OSError:
            return []

    def estado(self):
        rodando = self.rodando()
        nome = self.ativo or self.ultimo
        return {"rodando": rodando, "nome": nome, "progresso": self._progresso(nome) if nome else None,
                "log": self.log(nome, 6) if nome else [], "python": self.config["python"], "python_ok": self.python_ok}

    # ------------------------------------------------------------------ execucoes
    def _modelo_da_execucao(self, nome):
        pasta = os.path.join(EXECUCOES, nome)
        for arq in ("modelo_final.zip", "modelo_atual.zip"):
            if os.path.exists(os.path.join(pasta, arq)):
                return os.path.join(pasta, arq)
        ckpts = sorted(glob.glob(os.path.join(pasta, "checkpoint_*_steps.zip")),
                       key=lambda c: int(c.rsplit("_", 2)[-2]))
        if not ckpts:
            raise RuntimeError(f"{nome} ainda nao tem modelo salvo")
        return ckpts[-1]

    def politica(self, nome):
        return os.path.join(EXECUCOES, nome, "politica.npz")

    def listar(self):
        out = []
        for pasta in sorted(glob.glob(os.path.join(EXECUCOES, "*", "")), key=os.path.getmtime, reverse=True):
            nome = os.path.basename(os.path.dirname(pasta))
            p = self._progresso(nome) or {}
            tem_modelo = any(glob.glob(os.path.join(pasta, pad)) for pad in ("modelo_*.zip", "checkpoint_*.zip"))
            if not p and not tem_modelo and not os.path.exists(self.politica(nome)):   # importada: so o .npz
                continue
            out.append({"nome": nome, "passos": p.get("passos"), "total": p.get("total"),
                        "estado": "treinando" if nome == self.ativo and self.rodando() else p.get("estado", "-"),
                        "vx": p.get("vx"), "ep_len": p.get("ep_len"),
                        "politica": os.path.exists(self.politica(nome)), "modelo": tem_modelo,
                        "videos": sorted(os.path.basename(v) for v in glob.glob(os.path.join(pasta, "*.mp4"))),
                        "data": os.path.getmtime(pasta)})
        return out

    # ------------------------------------------------------------------ levar politicas para outro PC
    def pacote(self, nome):
        """Zip com politica.npz + config.json (o ambiente do treino) + info.json, para outro PC."""
        pasta = os.path.join(EXECUCOES, _nome_seguro(nome))
        if not os.path.exists(os.path.join(pasta, "politica.npz")):
            raise RuntimeError(f"{nome} nao tem politica.npz (exporte antes)")
        p = self._progresso(nome) or {}
        info = {"nome": nome, "passos": PoliticaNumpy(os.path.join(pasta, "politica.npz")).passos,
                "vx": p.get("vx"), "exportado": time.strftime("%Y-%m-%d %H:%M")}
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for arq in ("politica.npz", "config.json"):
                if os.path.exists(os.path.join(pasta, arq)):
                    z.write(os.path.join(pasta, arq), arq)
            z.writestr("info.json", json.dumps(info, indent=1))
        return buf.getvalue()

    def importar(self, nome_arquivo, dados, nome=None):
        """Recebe um .zip de pacote() ou um politica.npz solto; cria uma execucao nova com ele."""
        config = None
        if zipfile.is_zipfile(io.BytesIO(dados)) and not nome_arquivo.lower().endswith(".npz"):
            with zipfile.ZipFile(io.BytesIO(dados)) as z:
                nomes = {os.path.basename(n): n for n in z.namelist() if not n.endswith("/")}
                if "politica.npz" not in nomes:
                    raise RuntimeError("o zip nao tem politica.npz")
                if sum(i.file_size for i in z.infolist()) > 50e6:
                    raise RuntimeError("arquivo grande demais para uma politica")
                npz = z.read(nomes["politica.npz"])
                if "config.json" in nomes:
                    config = json.loads(z.read(nomes["config.json"]))
                if not nome and "info.json" in nomes:
                    nome = json.loads(z.read(nomes["info.json"])).get("nome")
        else:
            npz = dados
        if config is not None and not isinstance(config, dict):
            raise RuntimeError("config.json invalido")
        base = _nome_seguro(nome or os.path.splitext(os.path.basename(nome_arquivo))[0]) or "politica"
        final, i = base, 2
        while os.path.exists(os.path.join(EXECUCOES, final)):
            final, i = f"{base}_{i}", i + 1
        pasta = os.path.join(EXECUCOES, final)
        os.makedirs(pasta)
        try:
            with open(os.path.join(pasta, "politica.npz"), "wb") as fp:
                fp.write(npz)
            pol = PoliticaNumpy(os.path.join(pasta, "politica.npz"))   # confere se e mesmo uma politica
            entrada, saida = pol.camadas[0][0].shape[1], pol.camadas[-1][0].shape[0]
            if entrada not in (45, 47) or saida != 12:
                raise RuntimeError(f"politica de outro robo (entrada {entrada}, saida {saida})")
            if config is not None:
                with open(os.path.join(pasta, "config.json"), "w", encoding="utf-8") as fp:
                    json.dump(config, fp, indent=1)
        except Exception as e:
            shutil.rmtree(pasta, ignore_errors=True)
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError("nao e uma politica valida (precisa ser o politica.npz do exportar)") from e
        return final, config is not None

    def exportar(self, nome):
        """Gera politica.npz de uma execucao antiga (precisa do Python externo)."""
        py = self.python_ok or self.verificar_python()
        if not py:
            raise RuntimeError("precisa de um Python com torch e stable-baselines3 para exportar")
        modelo = self._modelo_da_execucao(nome)
        r = subprocess.run([py, os.path.join(self._codigo(), "exportar.py"), modelo, "--saida", self.politica(nome)],
                           capture_output=True, text=True, timeout=300, creationflags=_sem_janela())
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "falha ao exportar")
        return self.politica(nome)
