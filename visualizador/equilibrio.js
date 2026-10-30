// Painel do controle de equilibrio (controle/equilibrio.py roda no servidor).
const EIXOS = [
  ["pitch", "Frente / tras (pitch)"],
  ["roll", "Lateral (roll)"],
];
const GANHOS = [["kp", "Kp", 0.05], ["ki", "Ki", 0.05], ["kd", "Kd", 0.005], ["limite", "limite°", 1], ["zona_morta", "zona°", 0.1]];

export function iniciarEquilibrio({ nomes, enviar, online, aviso }) {
  const $ = (id) => document.getElementById(id);
  let cfg = null;
  let timer = null;

  // junta alteracoes num unico envio
  const pendente = {};
  function mandar(parcial) {
    mesclar(pendente, parcial);
    clearTimeout(timer);
    timer = setTimeout(() => {
      enviar({ tipo: "eq_config", config: JSON.parse(JSON.stringify(pendente)) });
      for (const k of Object.keys(pendente)) delete pendente[k];
    }, 250);
  }
  function mesclar(a, b) {
    for (const [k, v] of Object.entries(b)) {
      if (v && typeof v === "object" && !Array.isArray(v) && k !== "juntas") mesclar((a[k] ??= {}), v);
      else a[k] = v;
    }
  }

  function desenharEixos() {
    $("eqEixos").innerHTML = "";
    for (const [eixo, titulo] of EIXOS) {
      const e = cfg.eixos[eixo];
      const div = document.createElement("div");
      div.className = "eixo";
      div.innerHTML = `
        <label class="chk"><input type="checkbox" class="ativo"> <b style="color:var(--texto);font-weight:500">${titulo}</b></label>
        <div class="ganhos">${GANHOS.map(([k, rot, passo]) =>
          `<label>${rot}<input type="number" data-k="${k}" step="${passo}" min="0"></label>`).join("")}</div>
        <div class="barra" title="correcao atual"><i data-barra="${eixo}"></i></div>
        <div class="juntas"></div>
        <button class="add">+ junta</button>`;
      div.querySelector(".ativo").checked = e.ativo;
      div.querySelector(".ativo").onchange = (ev) => mandar({ eixos: { [eixo]: { ativo: ev.target.checked } } });
      div.querySelectorAll("input[data-k]").forEach((inp) => {
        inp.value = e[inp.dataset.k];
        inp.onchange = () => mandar({ eixos: { [eixo]: { [inp.dataset.k]: Math.max(0, +inp.value || 0) } } });
      });
      const lista = div.querySelector(".juntas");
      const juntas = Object.entries(e.juntas);
      const enviarJuntas = () => {
        const novo = {};
        lista.querySelectorAll(".jlinha").forEach((l) => {
          const n = l.querySelector("select").value;
          novo[n] = (novo[n] ?? 0) + (+l.querySelector("input").value || 0);
        });
        mandar({ eixos: { [eixo]: { juntas: novo } } });
      };
      const linha = (nome, peso) => {
        const l = document.createElement("div");
        l.className = "jlinha";
        l.innerHTML = `<select>${nomes.map((n) => `<option>${n}</option>`).join("")}</select>
          <input type="number" step="0.1" title="peso (negativo inverte o sentido)"><button title="remover">&times;</button>`;
        l.querySelector("select").value = nome;
        l.querySelector("input").value = peso;
        l.querySelector("select").onchange = enviarJuntas;
        l.querySelector("input").onchange = enviarJuntas;
        l.querySelector("button").onclick = () => { l.remove(); enviarJuntas(); };
        lista.appendChild(l);
      };
      juntas.forEach(([n, p]) => linha(n, p));
      div.querySelector(".add").onclick = () => {
        const livre = nomes.find((n) => !(n in e.juntas)) ?? nomes[0];
        linha(livre, 1);
        enviarJuntas();
      };
      $("eqEixos").appendChild(div);
    }
  }

  function preencher() {
    const s = cfg.sensor;
    $("eqAtivo").checked = cfg.ativo;
    $("eqFrente").value = s.frente;
    $("eqAlfa").value = s.alfa;
    $("eqInvPitch").checked = s.inverter_pitch;
    $("eqInvRoll").checked = s.inverter_roll;
    $("eqQueda").value = cfg.queda;
    $("eqTaxa").value = cfg.taxa_max;
    desenharEixos();
  }

  $("eqAtivo").onchange = (e) => enviar({ tipo: "eq_ativar", ligado: e.target.checked });
  $("eqNivelar").onclick = () => {
    if (!online()) return aviso("Conecte ao servidor e a ESP32 primeiro.");
    aviso("Nivelando: deixe o robo em pe e parado por 1 s...");
    enviar({ tipo: "eq_nivelar" });
  };
  $("eqFrente").onchange = (e) => {
    mandar({ sensor: { frente: e.target.value } });
    aviso("Clique em nivelar para aplicar o novo eixo.");
  };
  $("eqAlfa").onchange = (e) => mandar({ sensor: { alfa: Math.min(0.999, Math.max(0, +e.target.value)) } });
  $("eqInvPitch").onchange = (e) => mandar({ sensor: { inverter_pitch: e.target.checked } });
  $("eqInvRoll").onchange = (e) => mandar({ sensor: { inverter_roll: e.target.checked } });
  $("eqQueda").onchange = (e) => mandar({ queda: +e.target.value || 45 });
  $("eqTaxa").onchange = (e) => mandar({ taxa_max: +e.target.value || 300 });

  let ultimoSim = 0;
  function simular(forcar) {
    $("eqSimPitchV").textContent = $("eqSimPitch").value;
    $("eqSimRollV").textContent = $("eqSimRoll").value;
    const agora = performance.now();
    if (!forcar && agora - ultimoSim < 50) return;
    ultimoSim = agora;
    enviar({ tipo: "eq_simular", pitch: +$("eqSimPitch").value, roll: +$("eqSimRoll").value });
  }
  for (const id of ["eqSimPitch", "eqSimRoll"]) {
    $(id).oninput = () => simular(false);
    $(id).onchange = () => simular(true);
    $(id).ondblclick = () => { $(id).value = 0; simular(true); };
  }

  return {
    config(novo) {
      cfg = novo;
      // nao redesenha o que a pessoa esta editando agora
      if (!$("eqPainel").contains(document.activeElement)) preencher();
      else $("eqAtivo").checked = cfg.ativo;
    },
    estado(eq) {
      if (!eq) return;
      const st = $("eqStatus");
      st.classList.toggle("queda", eq.queda);
      if (!eq.fonte) st.textContent = eq.nivelado ? "sem IMU (conecte a ESP32 com firmware 2.2)" : "sem IMU · nao nivelado";
      else {
        const fonte = eq.fonte === "imu" ? "IMU" : "simulado";
        st.textContent = `${fonte} · pitch ${eq.pitch.toFixed(1)}° · roll ${eq.roll.toFixed(1)}°` +
          (eq.queda ? " · QUEDA: correcao parada" : eq.ativo ? ` · corrigindo ${eq.u.map((u) => u.toFixed(1)).join(" / ")}°` : "");
      }
      if (document.activeElement !== $("eqAtivo")) $("eqAtivo").checked = eq.ativo;
      EIXOS.forEach(([eixo], i) => {
        const barra = document.querySelector(`[data-barra="${eixo}"]`);
        const lim = cfg?.eixos[eixo].limite || 1;
        const f = Math.max(-1, Math.min(1, (eq.u[i] ?? 0) / lim));
        if (barra) Object.assign(barra.style, { left: `${50 + Math.min(0, f) * 50}%`, width: `${Math.abs(f) * 50}%` });
      });
    },
  };
}
