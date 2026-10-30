"""Biblioteca de poses e animacoes do usuario.

Poses de fabrica: config/poses.json (somente leitura, vem dentro do executavel).
Do usuario: <DADOS>/biblioteca.json  ({"poses": {nome: {junta: graus}}, "animacoes": {nome: anim}}).
Uma pose do usuario com o mesmo nome de uma de fabrica a substitui.
"""
import json
import os
import threading

from .modelo import DADOS, RAIZ

ARQUIVO = os.path.join(DADOS, "biblioteca.json")
_trava = threading.Lock()


def poses_fabrica():
    with open(os.path.join(RAIZ, "config", "poses.json"), encoding="utf-8") as fp:
        return {k: v for k, v in json.load(fp).items() if not k.startswith("_")}


def _ler_usuario():
    try:
        with open(ARQUIVO, encoding="utf-8") as fp:
            dados = json.load(fp)
    except (OSError, ValueError):
        dados = {}
    dados.setdefault("poses", {})
    dados.setdefault("animacoes", {})
    return dados


def _gravar(dados):
    os.makedirs(DADOS, exist_ok=True)
    tmp = ARQUIVO + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fp:
        json.dump(dados, fp, indent=1, ensure_ascii=False)
    os.replace(tmp, ARQUIVO)


def carregar():
    """{"poses": {nome: {"q": {...}, "fabrica": bool}}, "animacoes": {nome: anim}}"""
    usuario = _ler_usuario()
    poses = {n: {"q": q, "fabrica": True} for n, q in poses_fabrica().items()}
    poses.update({n: {"q": q, "fabrica": False} for n, q in usuario["poses"].items()})
    return {"poses": poses, "animacoes": usuario["animacoes"]}


def pose(nome):
    return carregar()["poses"][nome]["q"]


def _validar_nome(nome):
    nome = (nome or "").strip()
    if not nome or len(nome) > 60:
        raise ValueError("nome invalido (1 a 60 caracteres)")
    return nome


def salvar(tipo, nome, valor):
    """tipo = "poses" | "animacoes"."""
    nome = _validar_nome(nome)
    with _trava:
        dados = _ler_usuario()
        dados[tipo][nome] = valor
        _gravar(dados)


def apagar(tipo, nome):
    with _trava:
        dados = _ler_usuario()
        if nome not in dados[tipo]:
            raise ValueError("so da para apagar o que voce criou")
        del dados[tipo][nome]
        _gravar(dados)
