"""Emulador do firmware esp32_humanoide (porta "SIMULADO").

Responde o mesmo protocolo para testar o visualizador e o driver sem a placa.
Movimentos sao lineares no tempo (o firmware usa ease senoidal); sem IMU, sem BT.
Mantenha em sincronia com firmware/esp32_humanoide/src/main.cpp.
"""
import json
import math
import os
import threading
import time

from .modelo import DADOS

JUNTAS = ["quadril_roll_L", "quadril_pitch_L", "joelho_L", "tornozelo_L",
          "quadril_roll_R", "quadril_pitch_R", "joelho_R", "tornozelo_R",
          "ombro_L", "cotovelo_L", "ombro_R", "cotovelo_R"]
CFG_VEL = 60.0
ARQUIVO_NVS = os.path.join(DADOS, "esp32_simulada_nvs.json")


def _padrao():
    return [{"joint": None, "min": 0.0, "max": 180.0, "zero": 90.0, "dir": 1, "vel": 0.0, "en": 1} for _ in range(16)]


class _Canal:
    def __init__(self, a):
        self.ini = self.alvo = a
        self.t0 = 0.0
        self.dur = 0.0
        self.clamped = False

    def pos(self, t):
        if self.dur <= 0 or t - self.t0 >= self.dur:
            return self.alvo
        return self.ini + (self.alvo - self.ini) * (t - self.t0) / self.dur


class ESP32Simulada:
    def __init__(self):
        self.cfg = self._carregar() or _padrao()
        self.modo_cfg = False
        self.sujo = False
        self.canais = [_Canal(max(s["min"], min(s["max"], s["zero"]))) for s in self.cfg]
        self.saida = b""
        self.trava = threading.Lock()
        self.t_status = time.monotonic()
        self.imu_ms, self.t_imu = 0, 0.0
        self.inclinacao = (0.0, 0.0)   # pitch, roll (graus) de mentira, para testes
        self._emitir("READY esp32-humanoide v2.0 (simulado)")

    # interface de serial.Serial usada pelo driver
    def write(self, dados):
        for linha in dados.decode().splitlines():
            self._tratar(linha.strip().upper())

    def read(self, n):
        time.sleep(0.01)
        agora = time.monotonic()
        if agora - self.t_status >= 1.0:
            self.t_status = agora
            self._status()
        while self.imu_ms and agora - self.t_imu >= self.imu_ms / 1000:
            self.t_imu = agora if agora - self.t_imu > 1 else self.t_imu + self.imu_ms / 1000
            self._amostra_imu()
        with self.trava:
            dados, self.saida = self.saida[:n], self.saida[n:]
        return dados

    def close(self):
        pass

    # ---------------------------------------------------------------- internos
    def _emitir(self, linha):
        with self.trava:
            self.saida += (linha + "\n").encode()

    def _carregar(self):
        try:
            with open(ARQUIVO_NVS, encoding="utf-8") as fp:
                return json.load(fp)
        except (OSError, ValueError):
            return None

    def _amostra_imu(self):
        # placa montada como no robo real: "cima" = -Y do sensor, frente = +X do sensor
        p, r = (math.radians(a) for a in self.inclinacao)
        cima = (-math.sin(p), math.sin(r) * math.cos(p), math.cos(r) * math.cos(p))  # (frente, esq, cima) no corpo
        ax, ay, az = cima[0], -cima[2], cima[1]   # sensor: x = frente, y = baixo, z = esquerda
        self._emitir(f"I:{int(time.monotonic() * 1000)}:{ax:.3f}:{ay:.3f}:{az:.3f}:0.00:0.00:0.00")

    def _mover(self, ch, ang, ms):
        s, c = self.cfg[ch], self.canais[ch]
        if not s["en"]:
            return
        t = time.monotonic()
        ang = max(0.0, min(180.0, ang))
        if not self.modo_cfg:
            ang = max(s["min"], min(s["max"], ang))
        vel = min(CFG_VEL, s["vel"] or CFG_VEL) if self.modo_cfg else s["vel"]
        atual = c.pos(t)
        dur = ms / 1000
        if vel > 0:
            dur = max(dur, abs(ang - atual) / vel)
        elif dur <= 0.04:  # quadro de fluxo (JALL a 50 Hz): aplica direto
            dur = 0
        c.ini, c.alvo, c.t0, c.dur = atual, ang, t, dur

    def _mover_junta(self, ch, graus, ms):
        s = self.cfg[ch]
        servo = s["zero"] + s["dir"] * graus
        lim = max(s["min"], min(s["max"], servo))
        self.canais[ch].clamped = abs(lim - servo) > 0.01
        self._mover(ch, lim, ms)

    def _canal_da_junta(self, nome):
        return next((i for i, s in enumerate(self.cfg) if s["joint"] == nome), None)

    def _junta(self, tok):
        return next((j for j in JUNTAS if j.upper() == tok), None)

    def _status(self):
        t = time.monotonic()
        pos = [c.pos(t) for c in self.canais]
        st = {"t": int(t * 1000), "mode": "cfg" if self.modo_cfg else "run",
              "servos_busy": any(t - c.t0 < c.dur for c in self.canais),
              "channels": [round(p) for p in pos],
              "joints": {s["joint"]: round((pos[i] - s["zero"]) * s["dir"], 1)
                         for i, s in enumerate(self.cfg) if s["joint"]},
              "clamped": [i for i, c in enumerate(self.canais) if c.clamped],
              "imu": {"ok": False}}
        self._emitir(json.dumps(st, separators=(",", ":")))

    def _num(self, s):
        try:
            return float(s)
        except ValueError:
            return None

    def _tratar(self, linha):
        if not linha:
            return
        tok = linha.split(":")
        cmd, n = tok[0], len(tok)
        r = self._resposta(linha, tok, cmd, n)
        if r:
            self._emitir(r)

    def _resposta(self, linha, tok, cmd, n):
        t = time.monotonic()
        if linha == "PING":
            return "PONG"
        if linha == "STATUS":
            self._status()
            return "ACK"
        if linha == "CFG":
            servos = [{"ch": i, **s, "pos": round(self.canais[i].pos(t), 1)} for i, s in enumerate(self.cfg)]
            self._emitir(json.dumps({"mode": "cfg" if self.modo_cfg else "run", "dirty": self.sujo, "servos": servos},
                                    separators=(",", ":")))
            return "ACK"
        if linha == "JOINTS":
            self._emitir(",".join(JUNTAS))
            return "ACK"
        if linha == "INFO":
            self._emitir(f"INFO: fw=2.0 mode={'cfg' if self.modo_cfg else 'run'} bt=off (simulado)")
            return "ACK"
        if cmd == "IMUSTREAM" and n == 2 and tok[1].isdigit():
            self.imu_ms = int(tok[1])
            return "ACK"
        if linha in ("IMU",):
            self._emitir('{"ok":false}')
            return "ACK"
        if linha in ("SCAN", "ON", "OFF", "HELP"):
            return "ACK"
        if cmd == "MODE" and n == 2 and tok[1] in ("CFG", "RUN"):
            self.modo_cfg = tok[1] == "CFG"
            if not self.modo_cfg:
                for i in range(16):
                    self._mover(i, self.canais[i].pos(t), 0)
            return "ACK"
        if linha in ("REST", "GESTURE:REST"):
            for i, s in enumerate(self.cfg):
                self._mover(i, s["zero"], 1500)
            return "ACK"
        if linha == "CENTER":
            for i in range(16):
                self._mover(i, 90, 1000)
            return "ACK"
        if cmd in ("J", "JALL") and self.modo_cfg:
            return "ERR:IN_CFG_MODE"
        if cmd == "J":
            if n != 4 or self._num(tok[2]) is None:
                return "ERR:INVALID_CMD"
            j = self._junta(tok[1])
            if j is None:
                return "ERR:UNKNOWN_JOINT"
            ch = self._canal_da_junta(j)
            if ch is None:
                return "ERR:NOT_MAPPED"
            self._mover_junta(ch, float(tok[2]), int(tok[3]))
            return "ACK"
        if cmd == "JALL":
            if n != 2 + len(JUNTAS):
                return "ERR:INVALID_CMD"
            for j, v in zip(JUNTAS, tok[2:]):
                ch = self._canal_da_junta(j)
                if ch is not None and v != "X":
                    self._mover_junta(ch, float(v), int(tok[1]))
            return "ACK"

        # comandos com canal
        if n < 2 and cmd not in ("SAVE", "LOAD", "DEFAULTS"):
            return "ERR:INVALID_CMD"
        if cmd in ("SAVE", "LOAD", "DEFAULTS"):
            if not self.modo_cfg:
                return "ERR:NEED_CFG_MODE"
            if cmd == "SAVE":
                os.makedirs(DADOS, exist_ok=True)
                with open(ARQUIVO_NVS, "w", encoding="utf-8") as fp:
                    json.dump(self.cfg, fp, indent=1)
                self.sujo = False
            elif cmd == "LOAD":
                self.cfg = self._carregar() or _padrao()
                self.sujo = False
            else:
                self.cfg = _padrao()
                self.sujo = True
            return "ACK"
        if not tok[1].isdigit():
            return "ERR:INVALID_CMD"
        ch = int(tok[1])
        if ch > 15:
            return "ERR:OUT_OF_BOUNDS"
        s = self.cfg[ch]
        if cmd == "CENTER" and n == 2:
            self._mover(ch, 90, 1000)
            return "ACK"
        if cmd == "POS":
            ang = self._num(tok[2]) if n == 4 else None
            if ang is None:
                return "ERR:INVALID_CMD"
            if not 0 <= ang <= 180:
                return "ERR:OUT_OF_BOUNDS"
            if not s["en"]:
                return "ERR:DISABLED"
            if not self.modo_cfg and not s["min"] <= ang <= s["max"]:
                return f"ERR:LIMIT:{s['min']:.1f}:{s['max']:.1f}"
            self._mover(ch, ang, int(tok[3]))
            return "ACK"
        if cmd not in ("MAP", "LIM", "SETMIN", "SETMAX", "SETZERO", "ZERO", "DIR", "VEL", "EN", "WIGGLE"):
            return "ERR:INVALID_CMD"
        if not self.modo_cfg:
            return "ERR:NEED_CFG_MODE"
        atual = self.canais[ch].pos(t)
        if cmd == "MAP" and n == 3:
            j = None
            if tok[2] != "-":
                j = self._junta(tok[2])
                if j is None:
                    return "ERR:UNKNOWN_JOINT"
                antigo = self._canal_da_junta(j)
                if antigo is not None:
                    self.cfg[antigo]["joint"] = None
            s["joint"] = j
        elif cmd == "LIM" and n == 4:
            lo, hi = self._num(tok[2]), self._num(tok[3])
            if lo is None or hi is None:
                return "ERR:INVALID_CMD"
            if lo < 0 or hi > 180 or lo >= hi:
                return "ERR:OUT_OF_BOUNDS"
            s["min"], s["max"] = lo, hi
        elif cmd == "SETMIN" and n == 2:
            if atual >= s["max"]:
                return "ERR:OUT_OF_BOUNDS"
            s["min"] = round(atual, 1)
        elif cmd == "SETMAX" and n == 2:
            if atual <= s["min"]:
                return "ERR:OUT_OF_BOUNDS"
            s["max"] = round(atual, 1)
        elif cmd == "SETZERO" and n == 2:
            s["zero"] = round(atual, 1)
        elif cmd == "ZERO" and n == 3 and self._num(tok[2]) is not None:
            s["zero"] = float(tok[2])
        elif cmd == "DIR" and n == 3 and tok[2] in ("1", "-1"):
            s["dir"] = int(tok[2])
        elif cmd == "VEL" and n == 3 and (self._num(tok[2]) or 0) >= 0:
            s["vel"] = float(tok[2])
        elif cmd == "EN" and n == 3 and tok[2] in ("0", "1"):
            s["en"] = int(tok[2])
        elif cmd == "WIGGLE" and n == 2:
            return "ACK"
        else:
            return "ERR:INVALID_CMD"
        self.sujo = True
        return "ACK"
