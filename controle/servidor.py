"""Servidor de controle: roda controlador + simulador e transmite o estado ao visualizador.

    python -m controle.servidor                   # so simulacao; conecte a ESP32 pela pagina
    python -m controle.servidor --serial COM5     # ja conecta na ESP32 (USB ou COM do Bluetooth)
    python -m controle.servidor --serial SIMULADO # emulador do firmware, sem placa

Abra http://127.0.0.1:8790 no navegador.
"""
import argparse
import asyncio
import json
import time

import numpy as np
from aiohttp import WSMsgType, web

from . import biblioteca, hardware
from .controladores import CONTROLADORES
from .equilibrio import Equilibrio
from .modelo import RAIZ, Robo
from .simulador import SimuladorBaseFixa

TAXA_UI = 50  # Hz de envio ao visualizador e ao hardware


class Nucleo:
    def __init__(self, robo):
        self.robo = robo
        self.limite_modelo = robo.limite.copy()
        self.sim = SimuladorBaseFixa(robo)
        self.hw = None             # ESP32Humanoide conectada
        self.hw_seguir = False     # envia o comando de posicao aos servos
        self._q_hw, self._t_hw = None, 0.0   # ultimo quadro enviado aos servos
        self.controladores = {c.nome: c(robo) for c in CONTROLADORES}
        self.equilibrio = Equilibrio(robo)
        self._coleta_imu = None    # lista de amostras durante o "nivelar"
        self.clientes = set()
        self.trocar("manual")

    def trocar(self, nome):
        self.ctrl = self.controladores[nome]
        self.ctrl.reiniciar(self.sim.estado)

    def info(self):
        return {"tipo": "info", "controladores": list(self.controladores), "controlador": self.ctrl.nome,
                "limites": {n: np.degrees(l).round(1).tolist() for n, l in zip(self.robo.nomes, self.robo.limite)}}

    async def difundir(self, msg):
        texto = json.dumps(msg)
        for ws in list(self.clientes):
            try:
                await ws.send_str(texto)
            except ConnectionError:
                self.clientes.discard(ws)

    # ---------------------------------------------------------------- ESP32
    def aplicar_limites(self, config):
        """Limites gravados na ESP32 passam a valer no simulador e nos sliders."""
        self.robo.limite[:] = self.limite_modelo
        for nome, (lo, hi) in hardware.limites_das_juntas(config or {}).items():
            self.robo.limite[self.robo.indice[nome]] = np.radians([lo, hi])

    async def hw_conectar(self, porta):
        await self.hw_desconectar()
        self.hw = await asyncio.to_thread(hardware.ESP32Humanoide, porta)
        ok, linhas = await asyncio.to_thread(self.hw.comando, "JOINTS")
        juntas = next((l.split(",") for l in linhas if "," in l), [])
        if not ok or [j.lower() for j in juntas] != [n.lower() for n in self.robo.nomes]:
            await self.hw_desconectar()
            raise RuntimeError(f"firmware nao respondeu JOINTS com as 12 juntas do modelo: {linhas}")
        await self.hw_recarregar()
        ok, linhas = await asyncio.to_thread(self.hw.comando, "IMUSTREAM:20")   # IMU a 50 Hz
        if not ok:
            await self.difundir({"tipo": "aviso", "texto": "Firmware sem IMUSTREAM: grave a versao 2.2 para o equilibrio."})

    async def hw_desconectar(self):
        if self.hw:
            hw, self.hw = self.hw, None
            await asyncio.to_thread(hw.fechar)
        self.hw_seguir = False
        self.aplicar_limites(None)
        await self.difundir(self.info())

    async def hw_recarregar(self):
        config = await asyncio.to_thread(self.hw.ler_config)
        self.aplicar_limites(config)
        await self.difundir({"tipo": "hw_config", "config": config})
        await self.difundir(self.info())

    def hw_estado(self):
        if not self.hw:
            return None
        return {"porta": self.hw.porta, "modo": self.hw.modo, "seguir": self.hw_seguir, "status": self.hw.status,
                "log": self.hw.log[-8:]}

    async def laco_status(self):
        """Telemetria por consulta: o heartbeat da ESP32 so sai pela USB, nao pelo Bluetooth."""
        while True:
            await asyncio.sleep(1.0)
            hw = self.hw
            if hw:
                try:
                    await asyncio.to_thread(hw.atualizar_status)
                except Exception:  # porta caiu; o erro aparece no log da placa
                    pass

    # ---------------------------------------------------------------- laco
    def avancar(self, duracao, correcao=0.0):
        """Integra o simulador por `duracao` segundos; retorna o ultimo comando de posicao.
        `correcao` (rad) e somada ao comando de posicao (equilibrio)."""
        q_ref = None
        for _ in range(max(1, round(duracao / self.sim.dt))):
            e = self.sim.estado
            u = self.ctrl.calcular(e.t, e)
            if self.ctrl.tipo == "torque":
                self.sim.passo(tau=u)
            else:
                q_ref = self.robo.limitar(u + correcao)
                self.sim.passo(q_ref=q_ref)
        return q_ref

    async def laco(self):
        periodo = 1 / TAXA_UI
        proximo = time.perf_counter()
        while True:
            hw = self.hw
            for t_ms, acel, giro in (hw.amostras_imu() if hw else []):
                self.equilibrio.amostra(t_ms, acel, giro)
                if self._coleta_imu is not None:
                    self._coleta_imu.append((t_ms, acel, giro))
            q_ref = self.avancar(periodo, self.equilibrio.correcao(periodo))
            if hw and self.hw_seguir and hw.modo == "run":
                # servo de hobby so aceita posicao: controlador de torque manda o estado simulado
                q_hw = self.sim.estado.q if q_ref is None else q_ref
                agora = time.perf_counter()
                # manda quando muda (> 0.05 grau) e, parado, reenvia a cada 0.5 s
                if (self._q_hw is None or np.abs(q_hw - self._q_hw).max() > np.radians(0.05)
                        or agora - self._t_hw > 0.5):
                    await asyncio.to_thread(hw.enviar, q_hw, 1000 / TAXA_UI)
                    self._q_hw, self._t_hw = q_hw.copy(), agora
            else:
                self._q_hw = None
            e = self.sim.estado
            await self.difundir({
                "tipo": "estado", "t": e.t, "controlador": self.ctrl.nome,
                "fonte": "simulacao" + (" + servos" if hw and self.hw_seguir else ""),
                "q": self.robo.dict_de(e.q), "tau": self.robo.dict_de(e.tau, graus=False),
                "hw": self.hw_estado(),
                "anim": self.ctrl.estado_ui() if hasattr(self.ctrl, "estado_ui") else None,
                "eq": self.equilibrio.estado_ui(),
            })
            proximo += periodo
            await asyncio.sleep(max(0.0, proximo - time.perf_counter()))
            if time.perf_counter() - proximo > 0.5:  # atrasou demais: ressincroniza
                proximo = time.perf_counter()


# comandos que nao mudam a configuracao (nao precisam reler o CFG)
SO_MOVIMENTO = ("POS", "WIGGLE", "CENTER", "REST", "STATUS", "PING", "INFO", "IMU", "SCAN", "J:", "JALL")


async def tratar_hw(nucleo, ws, msg):
    tipo = msg["tipo"]
    try:
        if tipo == "hw_portas":
            desc = await asyncio.to_thread(hardware.descrever_portas)
            portas = [{"porta": p, "desc": desc.get(p, "emulador do firmware" if p == hardware.PORTA_SIMULADA else "")}
                      for p in await asyncio.to_thread(hardware.listar_portas)]
            await ws.send_json({"tipo": "hw_portas", "portas": portas})
        elif tipo == "hw_conectar":
            await nucleo.hw_conectar(msg["porta"])
        elif tipo == "hw_desconectar":
            await nucleo.hw_desconectar()
        elif tipo == "hw_seguir":
            if msg["ligado"] and nucleo.hw and not nucleo.hw_seguir:
                # leva os servos devagar ate a pose atual antes de comecar o fluxo de 50 Hz
                await asyncio.to_thread(nucleo.hw.enviar, nucleo.sim.estado.q, 1500)
                await asyncio.sleep(1.6)
            nucleo.hw_seguir = bool(msg["ligado"]) and nucleo.hw is not None
        elif tipo == "hw_cmd":
            if not nucleo.hw:
                raise RuntimeError("ESP32 nao conectada")
            cmd = msg["cmd"].strip()
            ok, linhas = await asyncio.to_thread(nucleo.hw.comando, cmd)
            await ws.send_json({"tipo": "hw_resp", "id": msg.get("id"), "cmd": cmd, "ok": ok, "linhas": linhas})
            if ok and not cmd.upper().startswith(SO_MOVIMENTO):
                await nucleo.hw_recarregar()
    except Exception as e:  # erro de porta/firmware vai para a pagina
        await ws.send_json({"tipo": "hw_resp", "id": msg.get("id"), "cmd": msg.get("cmd", tipo), "ok": False,
                            "linhas": [f"ERRO: {e}"]})


async def tratar_equilibrio(nucleo, ws, msg):
    eq, tipo = nucleo.equilibrio, msg["tipo"]
    try:
        if tipo == "eq_config":
            eq.configurar(msg["config"])
        elif tipo == "eq_ativar":
            eq.configurar({"ativo": bool(msg["ligado"])})
        elif tipo == "eq_simular":
            eq.simulado = None if msg.get("pitch") is None else (float(msg["pitch"]), float(msg["roll"]))
            if nucleo.hw and nucleo.hw.porta == hardware.PORTA_SIMULADA:   # emulador: inclina a IMU de mentira
                nucleo.hw.ser.inclinacao = eq.simulado or (0.0, 0.0)
            return
        elif tipo == "eq_nivelar":
            if not nucleo.hw:
                raise RuntimeError("conecte a ESP32 primeiro")
            nucleo._coleta_imu = []
            await asyncio.sleep(1.0)
            amostras, nucleo._coleta_imu = nucleo._coleta_imu, None
            eq.nivelar(amostras)
            await ws.send_json({"tipo": "aviso", "texto": f"Nivelado com {len(amostras)} amostras."})
        await nucleo.difundir({"tipo": "eq_config", "config": eq.cfg})
    except Exception as e:
        nucleo._coleta_imu = None
        await ws.send_json({"tipo": "aviso", "texto": f"equilibrio: {e}"})


OPERACOES_BIBLIOTECA = ("biblioteca", "salvar_pose", "apagar_pose", "salvar_anim", "apagar_anim")


async def tratar_biblioteca(nucleo, ws, msg):
    tipo = msg["tipo"]
    try:
        if tipo != "biblioteca":
            alvo = "poses" if tipo.endswith("pose") else "animacoes"
            if tipo.startswith("salvar"):
                await asyncio.to_thread(biblioteca.salvar, alvo, msg["nome"], msg["valor"])
            else:
                await asyncio.to_thread(biblioteca.apagar, alvo, msg["nome"])
        dados = await asyncio.to_thread(biblioteca.carregar)
        await (ws.send_json if tipo == "biblioteca" else nucleo.difundir)({"tipo": "biblioteca", **dados})
    except Exception as e:
        await ws.send_json({"tipo": "aviso", "texto": f"{tipo}: {e}"})


def criar_app(nucleo, porta_serial=None):
    async def ws_handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        nucleo.clientes.add(ws)
        await ws.send_json(nucleo.info())
        if nucleo.hw:
            await ws.send_json({"tipo": "hw_config", "config": nucleo.hw.config})
        await ws.send_json({"tipo": "eq_config", "config": nucleo.equilibrio.cfg})
        async for m in ws:
            if m.type != WSMsgType.TEXT:
                continue
            msg = json.loads(m.data)
            if msg["tipo"].startswith("hw_"):
                asyncio.create_task(tratar_hw(nucleo, ws, msg))
            elif msg["tipo"] == "controlador":
                nucleo.trocar(msg["nome"])
                await nucleo.difundir(nucleo.info())
            elif msg["tipo"].startswith("eq_"):
                asyncio.create_task(tratar_equilibrio(nucleo, ws, msg))
            elif msg["tipo"] in OPERACOES_BIBLIOTECA:
                await tratar_biblioteca(nucleo, ws, msg)
            else:
                # sliders/poses assumem o controle manual; anim_* liga o tocador de animacao
                destino = "animacao" if msg["tipo"].startswith("anim_") else None
                if msg["tipo"] in ("alvo", "pose") and nucleo.ctrl.nome in ("animacao", "agachamento", "acenar"):
                    destino = "manual"
                if destino and nucleo.ctrl.nome != destino:
                    nucleo.trocar(destino)
                    await nucleo.difundir(nucleo.info())
                nucleo.ctrl.comando(msg)
        nucleo.clientes.discard(ws)
        return ws

    async def iniciar(app):
        if porta_serial:
            await nucleo.hw_conectar(porta_serial)
        app["laco"] = asyncio.create_task(nucleo.laco())
        app["status"] = asyncio.create_task(nucleo.laco_status())

    async def parar(app):
        app["laco"].cancel()
        app["status"].cancel()
        await nucleo.hw_desconectar()

    app = web.Application()
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/", lambda r: web.HTTPFound("/visualizador/index.html"))
    app.router.add_get("/visualizador/", lambda r: web.HTTPFound("/visualizador/index.html"))
    for pasta in ("visualizador", "modelo", "config"):
        app.router.add_static(f"/{pasta}", f"{RAIZ}/{pasta}")
    app.on_startup.append(iniciar)
    app.on_cleanup.append(parar)
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--porta-http", type=int, default=8790)
    ap.add_argument("--serial", help="porta da ESP32 (ex.: COM5, ou SIMULADO); tambem da para conectar pela pagina")
    args = ap.parse_args()
    print(f"Visualizador: http://127.0.0.1:{args.porta_http}", flush=True)
    web.run_app(criar_app(Nucleo(Robo()), args.serial), host="127.0.0.1", port=args.porta_http, print=None)


if __name__ == "__main__":
    main()
