"""Programa desktop do humanoide: sobe o servidor de controle e abre o visualizador numa
janela propria (modo app do Edge/Chrome). Fecha sozinho quando a janela e fechada.

    python app_humanoide.py [--serial COM5]
Executavel: ferramentas/gerar_exe.py -> dist/Humanoide.exe
"""
import argparse
import asyncio
import logging
import os
import shutil
import socket
import subprocess
import sys
import urllib.request
import webbrowser

from aiohttp import web

from controle.modelo import DADOS, Robo
from controle.servidor import Nucleo, criar_app

PORTA_PADRAO = 8790
TITULO = "Humanoide 12 DOF"
ESPERA_FECHAR = 6      # s sem janela conectada -> encerra
ESPERA_ABRIR = 120     # s sem nenhuma janela ter conectado -> encerra

log = logging.getLogger("humanoide")


def porta_livre(porta):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", porta)) != 0


def ja_rodando(porta):
    """Outra instancia deste programa ja serve nessa porta?"""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{porta}/visualizador/index.html", timeout=1) as r:
            return TITULO in r.read().decode(errors="ignore")
    except OSError:
        return False


def porta_aleatoria():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def navegador_app():
    candidatos = [
        os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
        shutil.which("msedge") or "", shutil.which("chrome") or "",
    ]
    return next((c for c in candidatos if c and os.path.exists(c)), None)


def abrir_janela(url):
    exe = navegador_app()
    if not exe:
        webbrowser.open(url)
        return
    perfil = os.path.join(DADOS, "janela")   # perfil proprio: abre como janela separada
    subprocess.Popen([exe, f"--app={url}", f"--user-data-dir={perfil}", "--window-size=1400,900",
                      "--no-first-run", "--no-default-browser-check"])


async def rodar(args):
    porta = PORTA_PADRAO
    if not porta_livre(porta):
        if ja_rodando(porta):
            abrir_janela(f"http://127.0.0.1:{porta}/")
            return
        porta = porta_aleatoria()

    nucleo = Nucleo(Robo())
    runner = web.AppRunner(criar_app(nucleo, args.serial), access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", porta).start()
    url = f"http://127.0.0.1:{porta}/"
    log.info("servidor em %s", url)
    abrir_janela(url)

    # encerra quando a janela fecha (sem clientes WebSocket por ESPERA_FECHAR s)
    conectou, ocioso = False, 0.0
    while True:
        await asyncio.sleep(1)
        if nucleo.clientes:
            conectou, ocioso = True, 0.0
            continue
        ocioso += 1
        if (conectou and ocioso >= ESPERA_FECHAR) or (not conectou and ocioso >= ESPERA_ABRIR):
            break
    log.info("janela fechada, encerrando")
    await runner.cleanup()


def main():
    ap = argparse.ArgumentParser(description=TITULO)
    ap.add_argument("--serial", help="porta da ESP32 para conectar ao abrir (ex.: COM5)")
    args = ap.parse_args()
    os.makedirs(DADOS, exist_ok=True)
    handlers = [logging.FileHandler(os.path.join(DADOS, "humanoide.log"), encoding="utf-8")]
    if sys.stdout:   # no .exe sem console stdout e None
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", handlers=handlers)
    try:
        asyncio.run(rodar(args))
    except Exception:
        log.exception("erro fatal")
        raise


if __name__ == "__main__":
    main()
