"""Comunicacao com a ESP32 do humanoide (firmware/esp32_humanoide).

Transporte: USB Serial ou Bluetooth SPP (porta COM virtual), 115200 8N1, linhas ASCII.
A calibracao (canal <-> junta, limites, zero, sentido, velocidade) mora na ESP32 (NVS);
o host manda graus de junta (JALL) e a placa converte e satura nos limites.

    esp = ESP32Humanoide("COM5")      # ou "SIMULADO" para o emulador
    esp.comando("MODE:CFG")
    esp.comando("MAP:3:joelho_L")
    esp.enviar(q)                     # q em rad, ordem do robo.json
"""
import collections
import json
import queue
import threading
import time

import numpy as np

PORTA_SIMULADA = "SIMULADO"
RESPOSTAS_FINAIS = ("ACK", "PONG", "ERR:")


def listar_portas():
    try:
        from serial.tools import list_ports
    except ImportError:
        return [PORTA_SIMULADA]
    portas = [f"{p.device}" for p in sorted(list_ports.comports(), key=lambda p: p.device)]
    return portas + [PORTA_SIMULADA]


def descrever_portas():
    try:
        from serial.tools import list_ports
    except ImportError:
        return {}
    return {p.device: p.description for p in list_ports.comports()}


def limites_das_juntas(config):
    """CFG da ESP32 -> {junta: (min, max)} em graus de junta."""
    lim = {}
    for s in config.get("servos", []):
        if s["joint"]:
            a = (s["min"] - s["zero"]) * s["dir"]
            b = (s["max"] - s["zero"]) * s["dir"]
            lim[s["joint"]] = (min(a, b), max(a, b))
    return lim


class ESP32Humanoide:
    def __init__(self, porta, baud=115200):
        if porta == PORTA_SIMULADA:
            from .esp32_simulada import ESP32Simulada
            self.ser = ESP32Simulada()
        else:
            import serial  # pip install pyserial
            self.ser = serial.Serial(porta, baud, timeout=0.05)
        self.porta = porta
        self.status = None          # ultima telemetria (dict)
        self.t_status = 0.0         # time.monotonic() da ultima telemetria
        self.config = None          # ultimo CFG (dict)
        self.modo = None            # "run" | "cfg": vem do STATUS, do CFG e dos MODE:* aceitos
                                    # (pelo Bluetooth nao ha heartbeat; nao depender dele)
        self.log = []               # linhas soltas recebidas (READY, BT:, ERR de JALL...)
        self.imu = collections.deque(maxlen=500)   # amostras do IMUSTREAM: (t_ms, acel g (3), giro graus/s (3))
        self._resp = queue.Queue()
        self._trava = threading.Lock()
        self._em_comando = False
        self._rodando = True
        self._leitor = threading.Thread(target=self._ler, daemon=True)
        self._leitor.start()

    # ---------------------------------------------------------------- leitura
    def _ler(self):
        buf = b""
        while self._rodando:
            try:
                buf += self.ser.read(256)
            except Exception as e:  # porta caiu (cabo, BT)
                self.log.append(f"ERRO DE PORTA: {e}")
                self._rodando = False
                break
            while b"\n" in buf:
                linha, buf = buf.split(b"\n", 1)
                self._linha(linha.decode(errors="replace").strip())

    def _linha(self, linha):
        if not linha:
            return
        if linha.startswith("I:"):
            try:
                v = [float(x) for x in linha[2:].split(":")]
                self.imu.append((v[0], np.array(v[1:4]), np.array(v[4:7])))
            except (ValueError, IndexError):
                pass
            return
        if linha.startswith("{"):
            try:
                dados = json.loads(linha)
            except ValueError:
                dados = None
            if isinstance(dados, dict) and "servos_busy" in dados:
                self.status = dados           # heartbeat (so USB) ou resposta do STATUS
                self.t_status = time.monotonic()
                self.modo = dados.get("mode", self.modo)
                if self._em_comando != "STATUS":
                    return
            if isinstance(dados, dict) and "servos" in dados:
                self.config = dados
                self.modo = dados.get("mode", self.modo)
        if self._em_comando:
            self._resp.put(linha)
        elif linha != "ACK":  # ACKs do fluxo de JALL
            self.log.append(linha)
            del self.log[:-200]

    # ---------------------------------------------------------------- envio
    def comando(self, cmd, timeout=1.5):
        """Envia um comando e espera a resposta final (ACK/PONG/ERR). Retorna (ok, linhas)."""
        with self._trava:
            while not self._resp.empty():
                self._resp.get_nowait()
            self._em_comando = cmd.strip().upper() or True
            try:
                self.ser.write((cmd.strip() + "\n").encode())
                linhas = []
                fim = time.monotonic() + timeout
                while time.monotonic() < fim:
                    try:
                        linha = self._resp.get(timeout=fim - time.monotonic())
                    except queue.Empty:
                        break
                    linhas.append(linha)
                    if linha.startswith(RESPOSTAS_FINAIS):
                        ok = linha in ("ACK", "PONG")
                        if ok and self._em_comando in ("MODE:RUN", "MODE:CFG"):
                            self.modo = self._em_comando[5:].lower()
                        return ok, linhas
                return False, linhas + ["ERR:TIMEOUT"]
            finally:
                self._em_comando = False

    def atualizar_status(self, idade_max=1.5):
        """Pede STATUS se a telemetria estiver velha (Bluetooth nao recebe o heartbeat)."""
        if time.monotonic() - self.t_status > idade_max:
            self.comando("STATUS")

    def amostras_imu(self):
        """Retira as amostras novas da IMU (fila preenchida pela thread de leitura)."""
        out = []
        while self.imu:
            out.append(self.imu.popleft())
        return out

    def ler_config(self):
        ok, _ = self.comando("CFG")
        return self.config if ok else None

    def enviar(self, q, duracao_ms=0):
        """Posicao de todas as juntas (rad, ordem do robo.json; NaN = nao mexe), sem esperar ACK."""
        graus = ["X" if v is None or np.isnan(v) else f"{np.degrees(v):.1f}" for v in q]
        with self._trava:
            self.ser.write(f"JALL:{int(duracao_ms)}:{':'.join(graus)}\n".encode())

    def fechar(self):
        self._rodando = False
        self._leitor.join(timeout=1)
        self.ser.close()
