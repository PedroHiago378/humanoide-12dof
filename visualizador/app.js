// Visualizador do humanoide: arvore cinematica a partir de modelo/robo.json + malhas de robo.glb.
// Modo local: sliders movem o modelo. Com o servidor (controle/servidor.py) ligado, o estado
// vem por WebSocket (/ws) e os sliders podem enviar alvos ao controlador.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { iniciarAnimacao } from "./animacao.js";
import { iniciarEquilibrio } from "./equilibrio.js";

THREE.Object3D.DEFAULT_UP.set(0, 0, 1); // frame ROS: Z para cima

const RAD = Math.PI / 180;
const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- cena
const container = $("cena");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x14171c);
const camera = new THREE.PerspectiveCamera(40, 1, 0.005, 20);
camera.position.set(0.55, 0.38, 0.32);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0.13);
controls.enableDamping = true;

scene.add(new THREE.HemisphereLight(0xdde6ff, 0x30302a, 1.1));
const sol = new THREE.DirectionalLight(0xffffff, 1.6);
sol.position.set(0.6, 0.4, 1.2);
sol.castShadow = true;
sol.shadow.mapSize.set(2048, 2048);
Object.assign(sol.shadow.camera, { left: -0.4, right: 0.4, top: 0.4, bottom: -0.4, near: 0.1, far: 3 });
scene.add(sol);

const chao = new THREE.Mesh(new THREE.PlaneGeometry(3, 3), new THREE.ShadowMaterial({ opacity: 0.35 }));
chao.receiveShadow = true;
scene.add(chao);
const grade = new THREE.GridHelper(1, 20, 0x3a4250, 0x262b34);
grade.rotation.x = Math.PI / 2;
scene.add(grade);
scene.add(new THREE.AxesHelper(0.08));

function redimensionar() {
  const { clientWidth: w, clientHeight: h } = container;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(redimensionar).observe(container);

// ---------------------------------------------------------------- modelo
const [modelo, poses, gltf] = await Promise.all([
  fetch("../modelo/robo.json").then((r) => r.json()),
  fetch("../config/poses.json").then((r) => r.json()),
  new GLTFLoader().loadAsync("../modelo/robo.glb"),
]);
$("carregando").remove();

const base = new THREE.Group(); // pose flutuante do torso no mundo
scene.add(base);
base.position.set(0, 0, modelo.altura_base);

const elos = {};     // nome -> Group (frame do elo)
const juntas = {};   // nome -> { def, pivo, q, alvo }
for (const nome of Object.keys(modelo.elos)) {
  const g = new THREE.Group();
  g.name = nome;
  g.userData.def = modelo.elos[nome];
  elos[nome] = g;
}
base.add(elos[modelo.base]);
for (const j of modelo.juntas) {
  const pivo = new THREE.Group();
  pivo.position.fromArray(j.origem);
  elos[j.pai].add(pivo);
  pivo.add(elos[j.filho]);
  const eixo = new THREE.Vector3().fromArray(j.eixo).normalize();
  const marcador = new THREE.ArrowHelper(eixo, new THREE.Vector3(), 0.035, 0xffb000, 0.01, 0.006);
  marcador.userData.eixoJunta = true;
  pivo.add(marcador);
  juntas[j.nome] = { def: j, pivo, eixo, q: 0 };
}

// malhas do GLB: nos "<elo>__<i>" em coordenadas do proprio elo
const malhas = [];
gltf.scene.updateMatrixWorld(true);
const nos = [];
gltf.scene.traverse((o) => { if (o.isMesh) nos.push(o); });
for (const m of nos) {
  const nomeElo = (m.name || m.parent.name).split("__")[0];
  const elo = elos[nomeElo];
  if (!elo) continue;
  m.material = m.material.clone();
  m.castShadow = m.receiveShadow = true;
  m.position.set(0, 0, 0); m.quaternion.identity(); m.scale.set(1, 1, 1);
  elo.add(m);
  malhas.push(m);
}

// marcador do centro de massa + projecao no chao
const comBola = new THREE.Mesh(new THREE.SphereGeometry(0.006), new THREE.MeshBasicMaterial({ color: 0xff3366, depthTest: false }));
comBola.renderOrder = 10;
const comLinha = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
  new THREE.LineDashedMaterial({ color: 0xff3366, dashSize: 0.006, gapSize: 0.004, depthTest: false }));
const comSombra = new THREE.Mesh(new THREE.RingGeometry(0.006, 0.009, 24), new THREE.MeshBasicMaterial({ color: 0xff3366 }));
comSombra.position.z = 0.0005;
const grupoCom = new THREE.Group();
grupoCom.add(comBola, comLinha, comSombra);
scene.add(grupoCom);

// ---------------------------------------------------------------- estado
function aplicarJuntas() {
  for (const jt of Object.values(juntas)) jt.pivo.quaternion.setFromAxisAngle(jt.eixo, jt.q);
  base.updateMatrixWorld(true);
  if ($("vChao").checked && !estadoRemoto.base) {
    // apoia o pe mais baixo no chao (base so translada em Z)
    const caixa = new THREE.Box3();
    for (const lado of ["pe_L", "pe_R"]) caixa.union(new THREE.Box3().setFromObject(elos[lado]));
    base.position.z += -caixa.min.z;
    base.updateMatrixWorld(true);
  }
  atualizarCom();
}

const tmp = new THREE.Vector3();
let comMundo = new THREE.Vector3();
function atualizarCom() {
  comMundo.set(0, 0, 0);
  let m = 0;
  for (const elo of Object.values(elos)) {
    const d = elo.userData.def;
    tmp.fromArray(d.com).applyMatrix4(elo.matrixWorld);
    comMundo.addScaledVector(tmp, d.massa);
    m += d.massa;
  }
  comMundo.divideScalar(m);
  comBola.position.copy(comMundo);
  comLinha.geometry.setFromPoints([comMundo.clone(), new THREE.Vector3(comMundo.x, comMundo.y, 0)]);
  comLinha.computeLineDistances();
  comSombra.position.x = comMundo.x;
  comSombra.position.y = comMundo.y;
}

// ---------------------------------------------------------------- UI
$("resumo").textContent = `${modelo.juntas.length} juntas, ${Object.keys(modelo.elos).length} elos, ` +
  `${(modelo.massa_total * 1000).toFixed(0)} g`;

const ordemGrupos = [["Perna esquerda", "_L", /quadril|joelho|tornozelo/], ["Perna direita", "_R", /quadril|joelho|tornozelo/],
  ["Braco esquerdo", "_L", /ombro|cotovelo/], ["Braco direito", "_R", /ombro|cotovelo/]];
const ui = {};
for (const [titulo, lado, re] of ordemGrupos) {
  const sub = document.createElement("div");
  sub.className = "sub";
  sub.style.margin = "8px 0 2px";
  sub.textContent = titulo;
  $("juntas").appendChild(sub);
  for (const j of modelo.juntas.filter((j) => j.nome.endsWith(lado) && re.test(j.nome))) {
    const linha = document.createElement("div");
    linha.className = "junta";
    const [lo, hi] = j.limite.map((x) => Math.round(x / RAD));
    linha.innerHTML = `<span class="nome" title="${j.nome}">${j.nome.replace(/_[LR]$/, "")}</span>
      <input type="range" min="${lo}" max="${hi}" step="0.5" value="0"><span class="val">0.0</span>
      <div class="torque"><i></i></div>`;
    const input = linha.querySelector("input");
    input.addEventListener("input", () => {
      juntas[j.nome].q = input.value * RAD;
      linha.querySelector(".val").textContent = (+input.value).toFixed(1);
      if (comandando()) enviar({ tipo: "alvo", q: { [j.nome]: +input.value } });
      else aplicarJuntas();
    });
    input.addEventListener("pointerdown", () => { ui[j.nome].arrastando = true; });
    input.addEventListener("pointerup", () => { ui[j.nome].arrastando = false; });
    input.addEventListener("dblclick", () => { input.value = 0; input.dispatchEvent(new Event("input")); });
    linha.addEventListener("mouseenter", () => destacar(j.filho, true));
    linha.addEventListener("mouseleave", () => destacar(j.filho, false));
    ui[j.nome] = { input, val: linha.querySelector(".val"), torque: linha.querySelector(".torque i") };
    $("juntas").appendChild(linha);
  }
}

function destacar(nomeElo, on) {
  elos[nomeElo].traverse((o) => { if (o.isMesh) o.material.emissive?.setHex(on ? 0x1d4ed8 : 0x000000); });
}

function mostrarQ(qGraus) {
  for (const [nome, v] of Object.entries(qGraus)) {
    if (!juntas[nome]) continue;
    juntas[nome].q = v * RAD;
    if (ui[nome].arrastando) continue;
    ui[nome].input.value = v;
    ui[nome].val.textContent = (+v).toFixed(1);
  }
}

// poses, transicoes e editor de animacao (animacao.js)
const NOMES = modelo.juntas.map((j) => j.nome);
let avisoTimer = null;
function aviso(texto) {
  $("aviso").textContent = texto;
  $("aviso").style.display = "block";
  clearTimeout(avisoTimer);
  avisoTimer = setTimeout(() => { $("aviso").style.display = "none"; }, 3500);
}
const editor = iniciarAnimacao({
  nomes: NOMES,
  limites: (n) => [+ui[n].input.min, +ui[n].input.max],
  velMax: (n) => juntas[n].def.velocidade_max / RAD,
  getQ: () => Object.fromEntries(NOMES.map((n) => [n, Math.round(juntas[n].q / RAD * 10) / 10])),
  setQ: (q) => { mostrarQ(q); aplicarJuntas(); },
  enviar: (msg) => enviar(msg),
  online: () => ws?.readyState === 1,
  posesFabrica: Object.fromEntries(Object.entries(poses).filter(([k]) => !k.startsWith("_"))),
  aviso,
});
for (const [n, u] of Object.entries(ui)) {
  u.input.closest(".junta").querySelector(".nome").onclick = () => editor.selecionarJunta(n);
}
$("ltRecolher").onclick = () => document.body.classList.toggle("lt-fechada");
if (window.innerWidth < 760) document.body.classList.add("lt-fechada");   // tela estreita: comeca recolhida
const equilibrio = iniciarEquilibrio({ nomes: NOMES, enviar: (m) => enviar(m), online: () => ws?.readyState === 1, aviso });

$("vEixos").onchange = (e) => scene.traverse((o) => { if (o.userData.eixoJunta) o.visible = e.target.checked; });
$("vCom").onchange = (e) => { grupoCom.visible = e.target.checked; };
$("vChao").onchange = () => { base.position.z = modelo.altura_base; aplicarJuntas(); };
$("vTransp").onchange = (e) => malhas.forEach((m) => {
  m.material.transparent = e.target.checked;
  m.material.opacity = e.target.checked ? 0.35 : 1;
  m.material.depthWrite = !e.target.checked;
});

function atualizarInfo(extra = {}) {
  const linhas = {
    "COM (mm)": `${(comMundo.x * 1000).toFixed(1)}, ${(comMundo.y * 1000).toFixed(1)}, ${(comMundo.z * 1000).toFixed(1)}`,
    ...extra,
  };
  $("info").innerHTML = Object.entries(linhas).map(([k, v]) => `<b>${k}</b><span>${v}</span>`).join("");
}

// ---------------------------------------------------------------- WebSocket (servidor de controle)
let ws = null;
const estadoRemoto = { base: null };
const comandando = () => ws?.readyState === 1 && $("comandar").checked;
function enviar(msg) { if (ws?.readyState === 1) ws.send(JSON.stringify(msg)); }

function conectar() {
  if (location.protocol === "file:") return;
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onopen = () => {
    $("status").textContent = "conectado ao servidor";
    $("status").className = "on";
    $("comandar").disabled = $("controlador").disabled = false;
    $("comandar").checked = true;
    hwOnline(true);
    editor.conectou();
    document.body.classList.add("ws");
  };
  ws.onclose = () => {
    $("status").textContent = "offline (modo local)";
    $("status").className = "off";
    $("comandar").disabled = $("controlador").disabled = true;
    document.body.classList.remove("ws");
    estadoRemoto.base = null;
    hwOnline(false);
    setTimeout(conectar, 2000);
  };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.tipo === "info") {
      $("controlador").innerHTML = msg.controladores.map((c) => `<option>${c}</option>`).join("");
      $("controlador").value = msg.controlador;
      for (const [n, [lo, hi]] of Object.entries(msg.limites ?? {})) {
        Object.assign(ui[n].input, { min: lo, max: hi });
        ui[n].input.title = `limite ${lo} a ${hi} graus`;
      }
    } else if (msg.tipo === "eq_config") {
      equilibrio.config(msg.config);
    } else if (msg.tipo === "biblioteca") {
      editor.biblioteca(msg);
    } else if (msg.tipo === "aviso") {
      aviso(msg.texto);
    } else if (msg.tipo.startsWith("hw_")) {
      hwMensagem(msg);
    } else if (msg.tipo === "estado") {
      hwEstado(msg.hw);
      editor.pararLocal();
      editor.estado(msg.anim, msg.controlador);
      equilibrio.estado(msg.eq);
      // inclina o modelo como o robo real (IMU) ou o teste simulado
      if (!msg.base) {
        const eq = msg.eq;
        if (eq?.fonte) base.quaternion.setFromEuler(new THREE.Euler(eq.roll * RAD, eq.pitch * RAD, 0, "ZYX"));
        else base.quaternion.identity();
      }
      mostrarQ(msg.q);
      estadoRemoto.base = msg.base ?? null;
      if (msg.base) {
        base.position.fromArray(msg.base.p);
        base.quaternion.set(...msg.base.quat); // x, y, z, w
      }
      aplicarJuntas();
      for (const [n, tau] of Object.entries(msg.tau ?? {})) {
        const j = juntas[n].def;
        const f = Math.max(-1, Math.min(1, tau / j.torque_max));
        Object.assign(ui[n].torque.style, { left: `${50 + Math.min(0, f) * 50}%`, width: `${Math.abs(f) * 50}%`,
          background: Math.abs(f) > 0.9 ? "var(--erro)" : "var(--aviso)" });
      }
      const tauMax = Math.max(...Object.entries(msg.tau ?? {}).map(([n, t]) => Math.abs(t) / juntas[n].def.torque_max));
      atualizarInfo({ "t (s)": msg.t.toFixed(2), "Controlador": msg.controlador, "Fonte": msg.fonte,
        "Pior torque": isFinite(tauMax) ? `${(tauMax * 100).toFixed(0)}% do limite` : "-" });
    }
  };
}
$("controlador").onchange = (e) => enviar({ tipo: "controlador", nome: e.target.value });

// ---------------------------------------------------------------- ESP32: conexao e configuracao dos servos
const NOMES_JUNTAS = modelo.juntas.map((j) => j.nome);
let hwConectada = false;
let hwConfig = null;     // ultimo CFG da placa
let hwModo = "run";
let hwSeq = 0;
const hwPendentes = new Map();
const canalSel = () => +$("cfgCanal").value;

$("cfgCanal").innerHTML = Array.from({ length: 16 }, (_, i) => `<option value="${i}">${i}</option>`).join("");
$("cfgJunta").innerHTML = `<option value="-">(nenhuma)</option>` + NOMES_JUNTAS.map((n) => `<option>${n}</option>`).join("");

function hwLog(texto) {
  const el = $("hwLog");
  el.textContent = (el.textContent + texto + "\n").split("\n").slice(-40).join("\n");
  el.scrollTop = el.scrollHeight;
}

// envia um comando a ESP32 e devolve a resposta {ok, linhas}
function hwCmd(cmd) {
  const id = ++hwSeq;
  enviar({ tipo: "hw_cmd", id, cmd });
  return new Promise((res) => hwPendentes.set(id, res));
}

function hwOnline(on) {
  $("hwPorta").disabled = $("hwAtualizar").disabled = $("hwConectar").disabled = !on;
  if (on) enviar({ tipo: "hw_portas" });
  else hwEstado(null);
}

function hwHabilitar() {
  const cfg = hwConectada && hwModo === "cfg";
  $("hwSeguir").disabled = !hwConectada || cfg;
  $("cfgPainel").querySelectorAll("button, select, input").forEach((el) => {
    el.disabled = !hwConectada || (!cfg && !["modoRun", "modoCfg", "cfgCanal"].includes(el.id));
  });
  $("modoRun").classList.toggle("ativo", hwConectada && hwModo === "run");
  $("modoCfg").classList.toggle("ativo", hwConectada && hwModo === "cfg");
}

function hwEstado(hw) {
  const conectada = !!hw;
  if (conectada !== hwConectada) {
    hwConectada = conectada;
    $("hwConectar").textContent = conectada ? "desconectar" : "conectar";
    if (!conectada) { hwConfig = null; desenharTabela(); }
  }
  if (hw) {
    const st = hw.status ?? {};
    hwModo = hw.modo ?? st.mode ?? hwModo;
    const sat = (st.clamped ?? []).length ? ` · saturado: canal ${st.clamped.join(", ")}` : "";
    const imu = st.imu?.ok ? ` · IMU pitch ${st.imu.pitch} roll ${st.imu.roll}` : "";
    $("hwStatus").textContent = `${hw.porta} · modo ${hwModo.toUpperCase()}${imu}${sat}`;
    $("hwSeguir").checked = hw.seguir;
    for (const tr of $("cfgTabela").querySelectorAll("tr[data-ch]")) {
      tr.classList.toggle("sat", (st.clamped ?? []).includes(+tr.dataset.ch));
    }
  } else {
    $("hwStatus").textContent = "desconectada";
    $("hwSeguir").checked = false;
  }
  hwHabilitar();
}

function hwMensagem(msg) {
  if (msg.tipo === "hw_portas") {
    const atual = $("hwPorta").value;
    $("hwPorta").innerHTML = msg.portas.map((p) =>
      `<option value="${p.porta}">${p.porta}${p.desc ? " - " + p.desc : ""}</option>`).join("");
    if (msg.portas.some((p) => p.porta === atual)) $("hwPorta").value = atual;
  } else if (msg.tipo === "hw_config") {
    hwConfig = msg.config;
    if (hwConfig) hwModo = hwConfig.mode;
    desenharTabela();
    preencherCanal();
    hwHabilitar();
  } else if (msg.tipo === "hw_resp") {
    hwLog(`> ${msg.cmd}\n${msg.linhas.filter((l) => !l.startsWith("{")).join("\n")}`);
    hwPendentes.get(msg.id)?.(msg);
    hwPendentes.delete(msg.id);
  }
}

function desenharTabela() {
  const servos = hwConfig?.servos ?? [];
  $("cfgSujo").textContent = hwConfig?.dirty ? "alteracoes nao salvas" : "";
  $("cfgSujo").style.color = "var(--aviso)";
  $("cfgTabela").innerHTML = servos.length === 0 ? "" :
    `<tr><th>ch</th><th>junta</th><th>min</th><th>zero</th><th>max</th><th>dir</th><th>vel</th></tr>` +
    servos.map((s) => `<tr data-ch="${s.ch}" class="${s.ch === canalSel() ? "sel" : ""} ${s.en ? "" : "off"}">
      <td>${s.ch}</td><td>${s.joint ?? "-"}</td><td>${s.min.toFixed(0)}</td><td>${s.zero.toFixed(0)}</td>
      <td>${s.max.toFixed(0)}</td><td>${s.dir > 0 ? "+1" : "-1"}</td><td>${s.vel ? s.vel.toFixed(0) : "-"}</td></tr>`).join("");
  $("cfgTabela").querySelectorAll("tr[data-ch]").forEach((tr) => {
    tr.onclick = () => { $("cfgCanal").value = tr.dataset.ch; preencherCanal(); desenharTabela(); };
  });
}

let destacada = null;
function destacarJunta(nome) {
  if (destacada) destacar(juntas[destacada].def.filho, false);
  destacada = juntas[nome] ? nome : null;
  if (destacada) destacar(juntas[destacada].def.filho, true);
}

function preencherCanal() {
  const s = hwConfig?.servos?.[canalSel()];
  if (!s) return destacarJunta(null);
  $("cfgJunta").value = s.joint ?? "-";
  $("cfgDir").value = String(s.dir);
  $("cfgVel").value = s.vel;
  $("cfgEn").checked = !!s.en;
  $("cfgAng").value = s.pos;
  $("cfgAngVal").textContent = s.pos.toFixed(0);
  destacarJunta(s.joint);
}

$("hwAtualizar").onclick = () => enviar({ tipo: "hw_portas" });
$("hwConectar").onclick = () => {
  if (hwConectada) return enviar({ tipo: "hw_desconectar" });
  hwLog(`conectando em ${$("hwPorta").value}...`);
  enviar({ tipo: "hw_conectar", porta: $("hwPorta").value });
};
$("hwSeguir").onchange = (e) => enviar({ tipo: "hw_seguir", ligado: e.target.checked });
// ao entrar em CFG o envio de movimento para; ao voltar para RUN ele religa se estava ligado
let seguiaAntesCfg = false;
for (const b of [$("modoRun"), $("modoCfg")]) {
  b.onclick = async () => {
    if (b.dataset.modo === "CFG" && hwModo !== "cfg") {
      seguiaAntesCfg = $("hwSeguir").checked;
      enviar({ tipo: "hw_seguir", ligado: false });
    }
    const r = await hwCmd(`MODE:${b.dataset.modo}`);
    if (r.ok && b.dataset.modo === "RUN" && seguiaAntesCfg) {
      seguiaAntesCfg = false;
      enviar({ tipo: "hw_seguir", ligado: true });
    }
  };
}
$("cfgCanal").onchange = () => { preencherCanal(); desenharTabela(); };
$("cfgJunta").onchange = () => destacarJunta($("cfgJunta").value);
$("cfgWiggle").onclick = () => hwCmd(`WIGGLE:${canalSel()}`);
$("cfgEn").onchange = (e) => hwCmd(`EN:${canalSel()}:${e.target.checked ? 1 : 0}`);
$("cfgMap").onclick = () => hwCmd(`MAP:${canalSel()}:${$("cfgJunta").value}`);
$("cfgAplicar").onclick = async () => {
  await hwCmd(`DIR:${canalSel()}:${$("cfgDir").value}`);
  await hwCmd(`VEL:${canalSel()}:${Math.max(0, +$("cfgVel").value || 0)}`);
};
$("cfgPainel").querySelectorAll("button[data-cmd]").forEach((b) => {
  b.onclick = () => hwCmd(`${b.dataset.cmd}:${canalSel()}`);
});
$("cfgSalvar").onclick = () => hwCmd("SAVE");
$("cfgDescartar").onclick = () => hwCmd("LOAD");

// slider do angulo do servo: no maximo um POS a cada 80 ms
let posPendente = null;
let posTimer = null;
function enviarPos() {
  if (!posPendente) { posTimer = null; return; }
  hwCmd(posPendente);
  posPendente = null;
  posTimer = setTimeout(enviarPos, 80);
}
$("cfgAng").oninput = () => {
  $("cfgAngVal").textContent = $("cfgAng").value;
  posPendente = `POS:${canalSel()}:${$("cfgAng").value}:0`;
  if (!posTimer) enviarPos();
};

const esperar = (ms) => new Promise((r) => setTimeout(r, ms));
$("cfgTestar").onclick = async () => {
  const s = hwConfig?.servos?.[canalSel()];
  if (!s) return;
  let atual = s.pos;
  for (const alvo of [s.min, s.max, s.zero]) {
    const r = await hwCmd(`POS:${s.ch}:${alvo.toFixed(1)}:0`);
    if (!r.ok) return;
    $("cfgAng").value = alvo;
    $("cfgAngVal").textContent = alvo.toFixed(0);
    await esperar(Math.abs(alvo - atual) / Math.min(60, s.vel || 60) * 1000 + 400);
    atual = alvo;
  }
};
hwHabilitar();
conectar();

// ---------------------------------------------------------------- loop
redimensionar();
aplicarJuntas();
renderer.setAnimationLoop(() => {
  editor.quadro();
  controls.update();
  if (!ws || ws.readyState !== 1) atualizarInfo();
  renderer.render(scene, camera);
});
