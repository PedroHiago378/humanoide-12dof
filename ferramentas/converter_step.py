"""Converte o STEP do humanoide em modelo para visualizacao e controle.

Gera em humanoide/modelo/:
  robo.glb        malhas por elo (frame do elo, metros, Z para cima)
  robo.json       elos (massa, COM, inercia), juntas (origem, eixo, limites)
  robo.urdf       mesma arvore em URDF (para PyBullet / MuJoCo / ROS)
  meshes/*.stl    uma malha por elo (usada pelo URDF)

Convencao de eixos (ROS): X para frente, Y para a esquerda do robo, Z para cima.
No CAD original: X lateral (esquerda do robo = +X), Y para cima, Z para frente.

Uso:
  python ferramentas/converter_step.py C:/caminho/completo.step
"""
import argparse
import json
import os
import sys

import numpy as np
import trimesh
from OCP.BRep import BRep_Tool
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.GProp import GProp_GProps
from OCP.Quantity import Quantity_Color
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label, TDF_LabelSequence
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.XCAFDoc import XCAFDoc_ColorGen, XCAFDoc_ColorSurf, XCAFDoc_DocumentTool

AQUI = os.path.dirname(os.path.abspath(__file__))
SAIDA = os.path.join(AQUI, "..", "modelo")

# CAD (mm) -> ROS (m). Linhas: X_ros = Z_cad, Y_ros = X_cad, Z_ros = Y_cad
R_CAD = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
CENTRO_X_CAD = 25.0      # plano de simetria do robo (entre as pernas)
SOLA_Y_CAD = -132.0      # altura da sola no CAD

# ---------------------------------------------------------------------------
# Juntas, em coordenadas do CAD (mm). origem = ponto sobre o eixo do servo.
# eixo canonico: pitch = +X_cad (lateral), roll = +Z_cad (frente).
# eixo_servo = direcao do eixo de saida do servo no CAD; o sinal entre os dois
# vira "sentido_servo" para o mapeamento no hardware.
# ---------------------------------------------------------------------------
JUNTAS = [
    # nome,              pai,         filho,         origem CAD,             eixo,      eixo_servo
    ("quadril_roll_L",  "torso",     "quadril_L",   (50.0, 0.0, 0.0),       (0, 0, 1), (0, 0, 1)),
    ("quadril_pitch_L", "quadril_L", "coxa_L",      (61.8, -16.3, -3.4),    (1, 0, 0), (1, 0, 0)),
    ("joelho_L",        "coxa_L",    "canela_L",    (32.5, -64.9, -3.4),    (1, 0, 0), (-1, 0, 0)),
    ("tornozelo_L",     "canela_L",  "pe_L",        (48.9, -113.5, -3.5),   (1, 0, 0), (-1, 0, 0)),
    ("quadril_roll_R",  "torso",     "quadril_R",   (0.0, 0.0, 0.0),        (0, 0, 1), (0, 0, 1)),
    ("quadril_pitch_R", "quadril_R", "coxa_R",      (-11.8, -16.3, -3.4),   (1, 0, 0), (-1, 0, 0)),
    ("joelho_R",        "coxa_R",    "canela_R",    (17.5, -64.9, -3.5),    (1, 0, 0), (1, 0, 0)),
    ("tornozelo_R",     "canela_R",  "pe_R",        (-12.8, -113.5, -3.5),  (1, 0, 0), (-1, 0, 0)),
    ("ombro_L",         "torso",     "braco_L",     (50.0, 76.6, -5.5),     (1, 0, 0), (1, 0, 0)),
    ("cotovelo_L",      "braco_L",   "antebraco_L", (97.2, 76.6, 1.2),      (0, 0, 1), (0, 0, 1)),
    ("ombro_R",         "torso",     "braco_R",     (0.0, 76.6, -5.5),      (1, 0, 0), (-1, 0, 0)),
    ("cotovelo_R",      "braco_R",   "antebraco_R", (-47.2, 76.6, -12.3),   (0, 0, 1), (0, 0, -1)),
]
LIMITE_PADRAO = np.radians(90.0)
TORQUE_SERVO = 0.21       # N.m, MG90S a ~6 V (2.2 kgf.cm)
VELOCIDADE_SERVO = 10.5   # rad/s, MG90S ~0.1 s/60 graus

# Massas conhecidas (g): o volume do CAD vira so a distribuicao dentro do grupo.
MASSA_SERVO = 13.4
GRUPOS_MASSA = {"18650": 46.0, "esp32_wroom": 10.0, "PCA9685": 9.0, "ER-TFTM": 5.0}

CORES = {
    "servo": (0.13, 0.30, 0.70), "chifre": (0.92, 0.92, 0.90), "pcb": (0.10, 0.45, 0.20),
    "bateria": (0.85, 0.35, 0.10), "tela": (0.08, 0.08, 0.10), "impresso": (0.80, 0.82, 0.85),
}


def elo_da_peca(nomes, x_centro):
    """Decide a qual elo pertence uma peca folha, pelo caminho de nomes na montagem."""
    p = " / ".join(nomes)
    folha = nomes[-1]
    lado = "_L" if x_centro > CENTRO_X_CAD else "_R"
    eixo_servo = folha.startswith("axe") and "M90S_ali" in p

    if "Junta do Ombro" in p:
        if "junta basica completa v12" in p:
            if eixo_servo or any(s in p for s in ("/ Componente1:", "/ Componente2:", "MG90S_arm01")):
                return "antebraco" + lado
            return "braco" + lado
        if "Medium Horn" in p or "Componente2 (1)" in p or eixo_servo:
            return "braco" + lado
        return "torso"
    if "ponta bra" in p:
        return "antebraco" + lado
    if "Modelo Final Pernas" in p:
        if "Simplificados" in p:
            if "Componente2 (4)" in p or "MG90S_arm01" in p or eixo_servo:
                return "pe" + lado
            return "canela" + lado
        if "junta basica completa v5" in p:
            return ("coxa" if "Componente3 (3)" in p else "canela") + lado
        if "Componente5" in p or "MG90S horn v1:3" in p or "MG90S horn v1:4" in p:
            return "coxa" + lado
        if "M90S_ali v1 (6)" in p or "M90S_ali v1 (7)" in p:
            return ("coxa" if eixo_servo else "quadril") + lado
        if "M90S_ali v1 (4)" in p or "M90S_ali v1 (5)" in p:
            return "torso" if eixo_servo else "quadril" + lado
        if "Componente2 (2)" in p or "Componente4" in p:
            return "quadril" + lado
    return "torso"


def categoria(nomes):
    p = " / ".join(nomes)
    if "M90S_ali" in p:
        return "servo"
    if "horn" in p.lower() or "arm01" in p:
        return "chifre"
    if "18650" in p:
        return "bateria"
    if "ER-TFTM" in p:
        return "tela"
    if "esp32" in p or "PCA9685" in p:
        return "pcb"
    return "impresso"


def grupo_massa(nomes):
    """Chave do grupo cuja massa total e conhecida (None = densidade de impressao)."""
    for i, n in enumerate(nomes):
        if "M90S_ali" in n:
            return " / ".join(nomes[: i + 1]), MASSA_SERVO
        for chave, massa in GRUPOS_MASSA.items():
            if chave in n:
                return " / ".join(nomes[: i + 1]), massa
    return None, None


def ler_step(caminho):
    doc = TDocStd_Document(TCollection_ExtendedString("doc"))
    leitor = STEPCAFControl_Reader()
    leitor.SetNameMode(True)
    leitor.SetColorMode(True)
    if leitor.ReadFile(caminho) != 1:
        sys.exit(f"Falha ao ler {caminho}")
    leitor.Transfer(doc)
    return doc


def nome(label):
    a = TDataStd_Name()
    return a.Get().ToExtString() if label.FindAttribute(TDataStd_Name.GetID_s(), a) else "?"


def cor_xcaf(ct, labels):
    c = Quantity_Color()
    for l in labels:
        for tipo in (XCAFDoc_ColorSurf, XCAFDoc_ColorGen):
            if ct.GetColor_s(l, tipo, c):
                return (c.Red(), c.Green(), c.Blue())
    return None


def folhas(doc):
    """Itera (nomes do caminho, shape posicionado, cor) para cada peca folha."""
    st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    ct = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())

    def andar(label, caminho, loc, cor):
        ref = label
        if st.IsReference_s(label):
            loc = loc * st.GetLocation_s(label)
            ref = TDF_Label()
            st.GetReferredShape_s(label, ref)
        caminho = caminho + [nome(label)]
        cor = cor_xcaf(ct, (label, ref)) or cor
        filhos = TDF_LabelSequence()
        st.GetComponents_s(ref, filhos)
        if filhos.Length() == 0:
            yield caminho[1:], st.GetShape_s(ref), loc, cor
            return
        for i in range(1, filhos.Length() + 1):
            yield from andar(filhos.Value(i), caminho, loc, cor)

    raizes = TDF_LabelSequence()
    st.GetFreeShapes(raizes)
    for i in range(1, raizes.Length() + 1):
        yield from andar(raizes.Value(i), [], TopLoc_Location(), None)


def triangular(shape):
    """Vertices (mm, CAD global) e triangulos de um shape ja posicionado."""
    vs, fs, base = [], [], 0
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None:
            T = loc.Transformation()
            n = tri.NbNodes()
            for i in range(1, n + 1):
                p = tri.Node(i).Transformed(T)
                vs.append((p.X(), p.Y(), p.Z()))
            inv = face.Orientation() == TopAbs_REVERSED
            for i in range(1, tri.NbTriangles() + 1):
                a, b, c = tri.Triangle(i).Get()
                fs.append((base + a - 1, base + c - 1, base + b - 1) if inv else (base + a - 1, base + b - 1, base + c - 1))
            base += n
        exp.Next()
    return np.array(vs, float).reshape(-1, 3), np.array(fs, int).reshape(-1, 3)


def cad_para_ros(p_mm):
    """Ponto(s) do CAD (mm) para o frame do mundo ROS (m), origem no chao entre os pes."""
    p = np.asarray(p_mm, float) - np.array([CENTRO_X_CAD, SOLA_Y_CAD, 0.0])
    return (p @ R_CAD.T) / 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step")
    ap.add_argument("--densidade", type=float, default=0.8, help="g/cm3 efetiva das pecas impressas (PLA com preenchimento)")
    ap.add_argument("--deflexao", type=float, default=0.15, help="tolerancia da malha em mm")
    args = ap.parse_args()
    os.makedirs(os.path.join(SAIDA, "meshes"), exist_ok=True)

    print("Lendo STEP...")
    doc = ler_step(args.step)

    elos = {}         # nome -> {"malhas": {cor: [(v,f)]}, "pecas": [...]}
    grupos = {}       # chave -> [massa alvo, soma das massas a densidade 1]
    pecas = []
    for nomes, shape, loc, cor in folhas(doc):
        BRepMesh_IncrementalMesh(shape, args.deflexao, False, 0.35, True)  # pula se ja malhado
        s = shape.Moved(loc)
        v, f = triangular(s)
        if len(f) == 0:
            continue
        x_centro = (v[:, 0].min() + v[:, 0].max()) / 2
        elo = elo_da_peca(nomes, x_centro)
        cat = categoria(nomes)
        cor = tuple(round(c, 3) for c in (CORES[cat] if cat != "impresso" or cor is None else cor))
        e = elos.setdefault(elo, {"malhas": {}, "pecas": []})
        e["malhas"].setdefault(cor, []).append((v, f))

        gp = GProp_GProps()
        BRepGProp.VolumeProperties_s(s, gp)
        vol = abs(gp.Mass())  # mm3
        if vol < 1e-6:
            continue
        c = gp.CentreOfMass()
        M = gp.MatrixOfInertia()
        I = np.array([[M.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)])
        if gp.Mass() < 0:
            I = -I
        chave, alvo = grupo_massa(nomes)
        dens = 1.0 if chave else args.densidade * 1e-3  # g/mm3 (grupos escalados depois)
        if chave:
            grupos.setdefault(chave, [alvo, 0.0])[1] += vol
        peca = {"elo": elo, "grupo": chave, "m": vol * dens, "c": np.array([c.X(), c.Y(), c.Z()]), "I": I * dens}
        e["pecas"].append(peca)
        pecas.append(peca)

    for p in pecas:  # escala grupos com massa conhecida
        if p["grupo"]:
            alvo, soma = grupos[p["grupo"]]
            k = alvo / soma
            p["m"] *= k
            p["I"] = p["I"] * k

    origem_elo = {"torso": np.array([CENTRO_X_CAD, 0.0, 0.0])}  # torso: centro do quadril
    for j in JUNTAS:
        origem_elo[j[2]] = np.array(j[3], float)

    print(f"{'elo':14s} {'massa (g)':>10s}  pecas")
    saida_elos = {}
    cena = trimesh.Scene()
    for nome_elo in sorted(elos, key=lambda n: (n != "torso", n)):
        e = elos[nome_elo]
        o_ros = cad_para_ros(origem_elo[nome_elo])
        ps = e["pecas"]
        m = sum(p["m"] for p in ps)
        com = sum(p["m"] * p["c"] for p in ps) / m
        I = np.zeros((3, 3))
        for p in ps:
            d = p["c"] - com
            I += p["I"] + p["m"] * (d @ d * np.eye(3) - np.outer(d, d))
        I_ros = R_CAD @ I @ R_CAD.T * 1e-9   # g.mm2 -> kg.m2
        saida_elos[nome_elo] = {
            "origem": o_ros.round(5).tolist(),
            "massa": round(m / 1000.0, 5),
            "com": (cad_para_ros(com) - o_ros).round(5).tolist(),
            "inercia": I_ros.round(10).tolist(),
        }
        print(f"{nome_elo:14s} {m:10.1f}  {len(ps)}")

        partes = []
        for i, (cor, lista) in enumerate(e["malhas"].items()):
            v = np.vstack([cad_para_ros(a) - o_ros for a, _ in lista])
            offs = np.cumsum([0] + [len(a) for a, _ in lista[:-1]])
            f = np.vstack([b + k for (_, b), k in zip(lista, offs)])
            malha = trimesh.Trimesh(v, f, process=False)
            malha.merge_vertices()
            malha.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
                baseColorFactor=[*cor, 1.0], metallicFactor=0.1, roughnessFactor=0.6))
            cena.add_geometry(malha, node_name=f"{nome_elo}__{i}", geom_name=f"{nome_elo}__{i}")
            partes.append(malha)
        trimesh.util.concatenate(partes).export(os.path.join(SAIDA, "meshes", f"{nome_elo}.stl"))

    cena.export(os.path.join(SAIDA, "robo.glb"))

    saida_juntas = []
    for nome_j, pai, filho, origem, eixo, eixo_servo in JUNTAS:
        saida_juntas.append({
            "nome": nome_j, "pai": pai, "filho": filho,
            "origem": (cad_para_ros(origem) - cad_para_ros(origem_elo[pai])).round(5).tolist(),
            "eixo": (R_CAD @ np.array(eixo, float)).tolist(),
            "sentido_servo": int(np.sign(np.dot(eixo, eixo_servo))),
            "limite": [-LIMITE_PADRAO, LIMITE_PADRAO],
            "torque_max": TORQUE_SERVO, "velocidade_max": VELOCIDADE_SERVO,
        })
    modelo = {
        "nome": "humanoide_12dof",
        "frame": "X frente, Y esquerda, Z cima; metros, kg",
        "base": "torso",
        "altura_base": round(float(cad_para_ros(origem_elo["torso"])[2]), 5),
        "massa_total": round(sum(e["massa"] for e in saida_elos.values()), 5),
        "elos": saida_elos,
        "juntas": saida_juntas,
    }
    with open(os.path.join(SAIDA, "robo.json"), "w", encoding="utf-8") as fp:
        json.dump(modelo, fp, indent=2, ensure_ascii=False)
    escrever_urdf(modelo, os.path.join(SAIDA, "robo.urdf"))
    print(f"Massa total: {modelo['massa_total'] * 1000:.0f} g. Arquivos em {os.path.normpath(SAIDA)}")


def escrever_urdf(modelo, caminho):
    def v(a):
        return " ".join(f"{x:.6g}" for x in a)

    linhas = ['<?xml version="1.0"?>', f'<robot name="{modelo["nome"]}">']
    for nome_elo, e in modelo["elos"].items():
        I = e["inercia"]
        linhas += [
            f'  <link name="{nome_elo}">',
            f'    <inertial><origin xyz="{v(e["com"])}"/><mass value="{e["massa"]:.6g}"/>',
            f'      <inertia ixx="{I[0][0]:.6g}" ixy="{I[0][1]:.6g}" ixz="{I[0][2]:.6g}" '
            f'iyy="{I[1][1]:.6g}" iyz="{I[1][2]:.6g}" izz="{I[2][2]:.6g}"/></inertial>',
            f'    <visual><geometry><mesh filename="meshes/{nome_elo}.stl"/></geometry></visual>',
            f'    <collision><geometry><mesh filename="meshes/{nome_elo}.stl"/></geometry></collision>',
            "  </link>",
        ]
    for j in modelo["juntas"]:
        linhas += [
            f'  <joint name="{j["nome"]}" type="revolute">',
            f'    <parent link="{j["pai"]}"/><child link="{j["filho"]}"/>',
            f'    <origin xyz="{v(j["origem"])}"/><axis xyz="{v(j["eixo"])}"/>',
            f'    <limit lower="{j["limite"][0]:.4f}" upper="{j["limite"][1]:.4f}" '
            f'effort="{j["torque_max"]}" velocity="{j["velocidade_max"]}"/>',
            "  </joint>",
        ]
    linhas.append("</robot>")
    with open(caminho, "w", encoding="utf-8") as fp:
        fp.write("\n".join(linhas) + "\n")


if __name__ == "__main__":
    main()
