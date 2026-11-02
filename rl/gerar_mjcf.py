"""Gera o modelo MuJoCo (MJCF) do humanoide a partir de modelo/robo.json.

Saida em modelo/mujoco/:
  robo.xml         visual completo (malhas do CAD) + colisao, para ver/avaliar
  robo_treino.xml  so as cascas de colisao (carrega rapido nos ambientes de treino)
  colisao/*.stl    casca convexa de cada elo

Base livre (freejoint no torso), servos como atuadores de posicao com o torque do MG90S,
IMU (giroscopio, acelerometro, orientacao) no torso e sensores de toque nos pes.

    python rl/gerar_mjcf.py
"""
import json
import os

import numpy as np
import trimesh

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELO = os.path.join(RAIZ, "modelo")
SAIDA = os.path.join(MODELO, "mujoco")

# Servo MG90S (mesmos numeros do controle/simulador.py)
KP = 2.4            # N.m/rad
KV = 0.05           # N.m.s/rad
ARMADURA = 1e-3     # kg.m2, inercia do motor refletida (estimativa)
ATRITO = 0.01       # N.m, atrito seco da caixa de reducao
AMORTECIMENTO = 0.01

# Limites do robo real (graus): o joelho so dobra para tras (calibracao da ESP32)
LIMITES_REAIS = {"joelho_L": (0, 110), "joelho_R": (0, 110)}

PES = ("pe_L", "pe_R")
MAX_FACES = 190_000   # limite do leitor de STL do MuJoCo e 200 mil


def v(a, casas=6):
    return " ".join(f"{x:.{casas}g}" for x in a)


def cascas():
    """Casca convexa (colisao), caixa dos pes e malhas visuais (em partes de ate MAX_FACES)."""
    for sub in ("colisao", "visual"):
        os.makedirs(os.path.join(SAIDA, sub), exist_ok=True)
    caixas, partes = {}, {}
    for arq in sorted(os.listdir(os.path.join(MODELO, "meshes"))):
        nome = arq[:-4]
        malha = trimesh.load(os.path.join(MODELO, "meshes", arq))
        if nome in PES:
            caixas[nome] = malha.bounds
        malha.convex_hull.export(os.path.join(SAIDA, "colisao", arq))
        partes[nome] = []
        for i, ini in enumerate(range(0, len(malha.faces), MAX_FACES)):
            pedaco = malha.submesh([np.arange(ini, min(ini + MAX_FACES, len(malha.faces)))], append=True)
            pedaco.export(os.path.join(SAIDA, "visual", f"{nome}_{i}.stl"))
            partes[nome].append(f"{nome}_{i}")
    return caixas, partes


def corpo(d, nome, pos, filhos, visual, caixas, nivel):
    visual, partes = visual
    e = d["elos"][nome]
    I = np.array(e["inercia"])
    ind = "  " * nivel
    xml = [f'{ind}<body name="{nome}" pos="{v(pos)}">']
    if nome == d["base"]:
        xml += [f'{ind}  <freejoint name="base"/>',
                f'{ind}  <site name="imu" pos="0 0 0.06" size="0.004" group="4"/>']
    else:
        j = next(j for j in d["juntas"] if j["filho"] == nome)
        lo, hi = LIMITES_REAIS.get(j["nome"], np.degrees(j["limite"]))
        xml.append(f'{ind}  <joint name="{j["nome"]}" axis="{v(j["eixo"])}" range="{lo:g} {hi:g}"/>')
    xml.append(f'{ind}  <inertial pos="{v(e["com"])}" mass="{e["massa"]:.6g}" '
               f'fullinertia="{v([I[0,0], I[1,1], I[2,2], I[0,1], I[0,2], I[1,2]], 5)}"/>')
    if visual:
        xml += [f'{ind}  <geom class="visual" mesh="{p}"/>' for p in partes[nome]]
    if nome in caixas:
        lo, hi = caixas[nome]
        centro, meia = (lo + hi) / 2, (hi - lo) / 2
        xml.append(f'{ind}  <geom name="{nome}" class="pe" type="box" pos="{v(centro)}" size="{v(meia)}"/>')
        # sensor de toque: caixa invisivel (grupo 4) em volta do pe
        xml.append(f'{ind}  <site name="{nome}" pos="{v(centro)}" size="{v(meia + 0.001)}" type="box" group="4" rgba="0 0 0 0"/>')
    else:
        xml.append(f'{ind}  <geom name="{nome}" class="colisao" mesh="{nome}_col"/>')
    for j in filhos.get(nome, []):
        xml += corpo(d, j["filho"], j["origem"], filhos, (visual, partes), caixas, nivel + 1)
    xml.append(f"{ind}</body>")
    return xml


def gerar(visual, caixas, partes, d):
    filhos = {}
    for j in d["juntas"]:
        filhos.setdefault(j["pai"], []).append(j)
    malhas = [f'    <mesh name="{n}_col" file="colisao/{n}.stl"/>' for n in d["elos"]]
    if visual:
        malhas += [f'    <mesh name="{p}" file="visual/{p}.stl"/>' for ps in partes.values() for p in ps]
    atuadores = [
        f'    <position name="{j["nome"]}" joint="{j["nome"]}" kp="{KP}" kv="{KV}" '
        f'forcerange="{-j["torque_max"]} {j["torque_max"]}" ctrlrange="{v(np.radians(LIMITES_REAIS.get(j["nome"], np.degrees(j["limite"]))))}"/>'
        for j in d["juntas"]
    ]
    toque = [f'    <touch name="toque_{p}" site="{p}"/>' for p in PES]
    altura = d["altura_base"] + 0.002
    return f"""<mujoco model="{d['nome']}">
  <!-- gerado por rl/gerar_mjcf.py a partir de modelo/robo.json; nao edite a mao -->
  <compiler angle="degree" meshdir="." balanceinertia="true"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <default>
    <joint damping="{AMORTECIMENTO}" armature="{ARMADURA}" frictionloss="{ATRITO}"/>
    <geom contype="0" conaffinity="0"/>
    <default class="visual"><geom type="mesh" group="1" rgba="0.82 0.84 0.87 1" density="0"/></default>
    <default class="colisao"><geom type="mesh" group="3" contype="1" conaffinity="0" density="0" rgba="0.5 0.6 0.9 0.3"/></default>
    <default class="pe"><geom contype="1" conaffinity="0" density="0" friction="0.9 0.02 0.002" group="3" rgba="0.9 0.5 0.2 0.4"/></default>
  </default>
  <asset>
    <texture name="grade" type="2d" builtin="checker" rgb1="0.16 0.18 0.22" rgb2="0.12 0.13 0.16" width="512" height="512"/>
    <material name="chao" texture="grade" texrepeat="40 40" reflectance="0.05"/>
{chr(10).join(malhas)}
  </asset>
  <worldbody>
    <light pos="0.5 0.3 1.5" dir="-0.3 -0.2 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="chao" type="plane" size="20 20 0.1" material="chao" contype="0" conaffinity="1" friction="0.9 0.02 0.002"/>
{chr(10).join(corpo(d, d["base"], [0, 0, altura], filhos, (visual, partes), caixas, 2))}
  </worldbody>
  <actuator>
{chr(10).join(atuadores)}
  </actuator>
  <sensor>
    <gyro name="giro" site="imu"/>
    <accelerometer name="acel" site="imu"/>
    <framequat name="orientacao" objtype="site" objname="imu"/>
{chr(10).join(toque)}
  </sensor>
</mujoco>
"""


def main():
    with open(os.path.join(MODELO, "robo.json"), encoding="utf-8") as fp:
        d = json.load(fp)
    caixas, partes = cascas()
    for arq, visual in (("robo.xml", True), ("robo_treino.xml", False)):
        with open(os.path.join(SAIDA, arq), "w", encoding="utf-8") as fp:
            fp.write(gerar(visual, caixas, partes, d))
    print("Gerado em", SAIDA)


if __name__ == "__main__":
    main()
