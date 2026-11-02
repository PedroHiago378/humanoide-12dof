// Aba "Simulacao e treino": MuJoCo dentro do programa (controle/sim_mujoco.py) e treino de RL
// em processo separado (controle/rl_servico.py -> rl/treinar.py).
export function iniciarTreino({ enviar, online, aviso }) {
  const $ = (id) => document.getElementById(id);
  const css = getComputedStyle(document.documentElement);
  const cor = (v) => css.getPropertyValue(v).trim();
  let execucoes = [];
  let selecionada = null;
  let config = null;
  let padrao = {};
  let historico = [];

  // ------------------------------------------------------------------ abas
  document.querySelectorAll(".abas button").forEach((b) => {
    b.onclick = () => {
      document.querySelectorAll(".abas button").forEach((x) => x.classList.toggle("ativo", x === b));
      document.querySelectorAll(".aba").forEach((a) => { a.hidden = a.id !== b.dataset.aba; });
      if (b.dataset.aba === "abaSim") desenharGrafico();
    };
  });

  // ------------------------------------------------------------------ simulacao
  const exigeServidor = () => (online() ? true : (aviso("Precisa do servidor (abra pelo programa)."), false));
  let ultimaTroca = 0;
  $("simLigar").onchange = (e) => {
    ultimaTroca = performance.now();
    if (!exigeServidor()) { e.target.checked = false; return; }
    enviar({ tipo: "sim_ligar", ligado: e.target.checked });
    if (e.target.checked) {
      enviar({ tipo: "sim_modo", modo: $("simModo").value });
      enviar({ tipo: "sim_config", ...comando(), massa: +$("simMassa").value, empurroes: $("simEmpurroes").checked });
      enviar({ tipo: "sim_politica", nome: $("simPolitica").value || null });
      if ($("simPoliticaGiro").value) enviar({ tipo: "sim_politica_giro", nome: $("simPoliticaGiro").value });
    }
  };
  $("simModo").onchange = () => {
    if (online()) enviar({ tipo: "sim_modo", modo: $("simModo").value });
    if ($("simModo").value === "poses") aviso("Use os sliders, poses e animacoes da aba Controle (com \"comandar\" marcado).");
  };
  $("simPolitica").onchange = () => {
    ultimaTroca = performance.now();
    if (!exigeServidor()) return;
    if ($("simPolitica").value) aviso("Carregando politica...");
    enviar({ tipo: "sim_politica", nome: $("simPolitica").value || null });
  };
  $("simPoliticaGiro").onchange = () => {
    ultimaTroca = performance.now();
    if (!exigeServidor()) return;
    enviar({ tipo: "sim_politica_giro", nome: $("simPoliticaGiro").value || null });
  };
  $("simRecarregar").onclick = () => { $("simPolitica").onchange(); $("simPoliticaGiro").onchange(); };
  // comando: frente (cm/s), lado (cm/s), giro (graus/s) -> m/s e rad/s
  const comando = () => ({ vx: +$("simVx").value / 100, vy: +$("simVy").value / 100, giro: (+$("simGiro").value * Math.PI) / 180 });
  let ultimoCmd = 0;
  function mandarComando(forcar) {
    $("simVxV").textContent = `${$("simVx").value} cm/s`;
    $("simVyV").textContent = `${$("simVy").value} cm/s`;
    $("simGiroV").textContent = `${$("simGiro").value} °/s`;
    const agora = performance.now();
    if (!forcar && agora - ultimoCmd < 60) return;
    ultimoCmd = agora;
    if (online()) enviar({ tipo: "sim_config", ...comando() });
  }
  for (const id of ["simVx", "simVy", "simGiro"]) {
    $(id).oninput = () => mandarComando(false);
    $(id).onchange = () => mandarComando(true);
    $(id).ondblclick = () => { $(id).value = 0; mandarComando(true); };
  }
  // teclado: segurar a tecla move naquele eixo; soltar volta para zero
  const TECLAS = { KeyW: ["simVx", 1], KeyS: ["simVx", -1], KeyA: ["simVy", 1], KeyD: ["simVy", -1],
    KeyQ: ["simGiro", 1], KeyE: ["simGiro", -1] };
  window.addEventListener("keydown", (e) => {
    const t = TECLAS[e.code];
    if (!t || e.repeat || !$("simLigar").checked || $("abaSim").hidden) return;
    if (/INPUT|SELECT|TEXTAREA/.test(document.activeElement?.tagName) && document.activeElement.type !== "range") return;
    e.preventDefault();
    const el = $(t[0]);
    el.value = t[1] > 0 ? el.max : el.min;
    mandarComando(true);
  });
  window.addEventListener("keyup", (e) => {
    const t = TECLAS[e.code];
    if (!t || !$("simLigar").checked) return;
    $(t[0]).value = 0;
    mandarComando(true);
  });
  $("simMassa").oninput = () => { $("simMassaV").textContent = `${(+$("simMassa").value).toFixed(2)}x`; };
  $("simMassa").onchange = () => enviar({ tipo: "sim_config", massa: +$("simMassa").value });
  $("simEmpurroes").onchange = (e) => enviar({ tipo: "sim_config", empurroes: e.target.checked });
  $("simEmpurrar").onclick = () => enviar({ tipo: "sim_empurrar", forca: 2.0 });
  $("simReiniciar").onclick = () => enviar({ tipo: "sim_reiniciar" });

  // a ultima politica escolhida fica salva no navegador e volta selecionada ao abrir o programa
  const lembrar = (id, v) => { try { localStorage.setItem(id, v ?? ""); } catch {} };
  const lembrada = (id) => { try { return localStorage.getItem(id) ?? ""; } catch { return ""; } };
  for (const id of ["simPolitica", "simPoliticaGiro", "simModo"]) $(id).addEventListener("change", () => lembrar(id, $(id).value));
  if (lembrada("simModo")) $("simModo").value = lembrada("simModo");

  function desenharPoliticas() {
    const opcoes = execucoes.filter((e) => e.modelo || e.politica);
    const itens = opcoes.map((e) => `<option value="${e.nome}">${e.nome}${e.politica ? "" : " (exportar)"}${e.estado === "treinando" ? " · treinando" : ""}</option>`).join("");
    for (const [id, vazio] of [["simPolitica", "(sem politica: pose padrao)"], ["simPoliticaGiro", "(nenhuma: usa a de caminhada)"]]) {
      const atual = $(id).options.length ? $(id).value : lembrada(id);
      $(id).innerHTML = `<option value="">${vazio}</option>` + itens;
      if (opcoes.some((e) => e.nome === atual)) $(id).value = atual;
    }
  }

  // a simulacao pode ter sido ligada por outra janela: a aba mostra o que o servidor esta rodando
  function sincronizar(mj) {
    if (performance.now() - ultimaTroca < 1500) return;   // quadros que ja estavam a caminho
    $("simLigar").checked = true;
    for (const [id, v] of [["simPolitica", mj.politica], ["simPoliticaGiro", mj.politica_giro]]) {
      if (document.activeElement === $(id) || $(id).value === (v ?? "")) continue;
      if ([...$(id).options].some((o) => o.value === (v ?? ""))) { $(id).value = v ?? ""; lembrar(id, v); }
    }
  }

  // ------------------------------------------------------------------ treino
  const fmtM = (n) => (n == null ? "-" : n >= 1e6 ? `${(n / 1e6).toFixed(2)} M` : `${Math.round(n / 1e3)} k`);
  const fmtTempo = (s) => (s > 3600 ? `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min` : `${Math.round(s / 60)} min`);

  $("rlIniciar").onclick = () => {
    if (!exigeServidor()) return;
    enviar({ tipo: "rl_iniciar", nome: $("rlNome").value.trim() || null,
      passos: Math.round(+$("rlPassos").value * 1e6), envs: +$("rlEnvs").value });
  };
  $("rlContinuar").onclick = () => {
    if (!selecionada) return aviso("Selecione uma execucao na tabela.");
    if (!confirm(`Continuar o treino de "${selecionada}" por mais ${$("rlPassos").value} M passos?`)) return;
    enviar({ tipo: "rl_iniciar", nome: selecionada, continuar: selecionada,
      passos: Math.round(+$("rlPassos").value * 1e6), envs: +$("rlEnvs").value });
  };
  $("rlParar").onclick = () => { if (confirm("Parar o treino? Ele salva o que aprendeu.")) enviar({ tipo: "rl_parar" }); };
  $("rlRestaurar").onclick = () => enviar({ tipo: "rl_restaurar" });
  $("rlVerificar").onclick = () => { aviso("Procurando Python com torch..."); enviar({ tipo: "rl_python", python: $("rlPython").value.trim() }); };
  for (const id of ["rlPassos", "rlEnvs"]) {
    $(id).onchange = () => enviar({ tipo: "rl_config", config: { passos: Math.round(+$("rlPassos").value * 1e6), envs: +$("rlEnvs").value } });
  }

  function desenharRecompensas() {
    const painel = $("rlRecompensas");
    if (painel.contains(document.activeElement)) return;
    painel.innerHTML = "";
    for (const [k, v] of Object.entries(config.recompensas)) {
      const rot = document.createElement("span");
      rot.textContent = k.replace(/_/g, " ");
      rot.title = `padrao: ${padrao[k]}`;
      rot.classList.toggle("mudou", padrao[k] !== v);
      const inp = Object.assign(document.createElement("input"), { type: "number", step: "0.05", value: v });
      inp.onchange = () => enviar({ tipo: "rl_config", config: { recompensas: { [k]: +inp.value } } });
      painel.append(rot, inp);
    }
  }

  function desenharExecucoes() {
    $("rlExecucoes").innerHTML = execucoes.length === 0 ? "<tr><td class='sub'>nenhum treino ainda</td></tr>" :
      `<tr><th>execucao</th><th>passos</th><th>vel.</th><th></th></tr>` +
      execucoes.map((e) => `<tr data-nome="${e.nome}" class="${e.nome === selecionada ? "sel" : ""}">
        <td title="${new Date(e.data * 1000).toLocaleString()}">${e.nome}<br><span class="sub">${e.estado}</span></td>
        <td>${fmtM(e.passos)}</td><td>${e.vx == null ? "-" : (e.vx * 100).toFixed(1) + " cm/s"}</td>
        <td><button data-acao="simular">simular</button>${(e.videos ?? []).map((v) =>
          `<button data-video="${encodeURIComponent(e.nome)}/${encodeURIComponent(v)}" title="${v}">video</button>`).join("")}</td></tr>`).join("");
    $("rlExecucoes").querySelectorAll("tr[data-nome]").forEach((tr) => {
      tr.onclick = (ev) => {
        selecionada = tr.dataset.nome;
        desenharExecucoes();
        if (ev.target.dataset.video) return window.open(`/execucoes/${ev.target.dataset.video}`, "_blank");
        if (ev.target.dataset.acao === "simular") {
          $("simPolitica").value = selecionada;
          if (!$("simLigar").checked) { $("simLigar").checked = true; $("simLigar").onchange({ target: $("simLigar") }); }
          else $("simPolitica").onchange();
        }
      };
    });
  }

  // ------------------------------------------------------------------ aba Politicas: exportar / receber
  function desenharListaPoliticas() {
    const lista = execucoes.filter((e) => e.politica || e.modelo);
    $("polLista").innerHTML = lista.length === 0 ? "<tr><td class='sub'>nenhuma politica ainda</td></tr>" :
      `<tr><th>politica</th><th>passos</th><th></th></tr>` + lista.map((e) => `<tr>
        <td>${e.nome}<br><span class="sub">${new Date(e.data * 1000).toLocaleDateString()}</span></td><td>${fmtM(e.passos)}</td>
        <td>${e.politica ? `<a href="/politicas/${encodeURIComponent(e.nome)}.zip" download="${e.nome}.zip"><button>exportar .zip</button></a>`
          : `<button data-gerar="${e.nome}" title="Gera o politica.npz a partir do modelo (precisa do Python do treino)">gerar .npz</button>`}</td></tr>`).join("");
    $("polLista").querySelectorAll("button[data-gerar]").forEach((b) => {
      b.onclick = () => { aviso("Gerando politica.npz..."); enviar({ tipo: "rl_exportar", nome: b.dataset.gerar }); };
    });
  }

  async function receber(arquivo) {
    if (!arquivo) return;
    if (!exigeServidor()) return;
    $("polStatus").textContent = `enviando ${arquivo.name}...`;
    const form = new FormData();
    form.append("arquivo", arquivo);
    if ($("polNome").value.trim()) form.append("nome", $("polNome").value.trim());
    try {
      const r = await fetch("/politicas/importar", { method: "POST", body: form });
      const j = await r.json();
      if (!r.ok) throw new Error(j.erro || r.statusText);
      $("polStatus").textContent = `recebida como "${j.nome}"${j.config ? "" : " (sem config.json: ambiente padrao)"}. Ja aparece na aba Simulacao.`;
      $("polNome").value = "";
    } catch (e) {
      $("polStatus").textContent = `nao recebida: ${e.message}`;
    }
    $("polArquivo").value = "";
  }
  $("polSoltar").onclick = () => $("polArquivo").click();
  $("polSoltar").onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") $("polArquivo").click(); };
  $("polArquivo").onchange = () => receber($("polArquivo").files[0]);
  $("polSoltar").ondragover = (e) => { e.preventDefault(); $("polSoltar").classList.add("sobre"); };
  $("polSoltar").ondragleave = () => $("polSoltar").classList.remove("sobre");
  $("polSoltar").ondrop = (e) => {
    e.preventDefault();
    $("polSoltar").classList.remove("sobre");
    receber(e.dataTransfer.files[0]);
  };

  function desenharGrafico() {
    const c = $("rlGrafico");
    const r = c.getBoundingClientRect();
    if (!r.width) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = r.width * dpr;
    c.height = r.height * dpr;
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, r.width, r.height);
    if (historico.length < 2) {
      g.fillStyle = cor("--fraco");
      g.font = "11px system-ui";
      g.fillText("o grafico aparece quando o treino comeca", 8, r.height / 2);
      return;
    }
    const x0 = historico[0].passos, x1 = historico.at(-1).passos || 1;
    const serie = (campo, escala, max, corLinha) => {
      g.strokeStyle = corLinha;
      g.lineWidth = 1.5;
      g.beginPath();
      historico.forEach((h, i) => {
        const v = (h[campo] ?? 0) * escala;
        const x = 4 + ((h.passos - x0) / (x1 - x0 || 1)) * (r.width - 8);
        const y = r.height - 4 - (Math.min(v, max) / max) * (r.height - 16);
        i ? g.lineTo(x, y) : g.moveTo(x, y);
      });
      g.stroke();
    };
    const maxLen = Math.max(...historico.map((h) => (h.ep_len ?? 0) * 0.02), 1);
    const maxVx = Math.max(...historico.map((h) => (h.vx ?? 0) * 100), 1);
    serie("ep_len", 0.02, maxLen, cor("--acento"));
    serie("vx", 100, maxVx, "#ffb000");
    g.fillStyle = cor("--fraco");
    g.font = "10px system-ui";
    g.fillText(`episodio ate ${maxLen.toFixed(0)} s`, 6, 11);
    g.fillStyle = "#ffb000";
    g.fillText(`vel. ate ${maxVx.toFixed(1)} cm/s`, r.width - 100, 11);
  }

  function mostrarTreino(st) {
    const p = st.progresso;
    $("rlIniciar").disabled = $("rlContinuar").disabled = st.rodando;
    $("rlParar").disabled = !st.rodando;
    $("rlStatus").textContent = st.rodando ? `treinando ${st.nome}` : (p ? `ultimo: ${st.nome} (${p.estado})` : "sem treino rodando");
    if (p) {
      const frac = Math.min(1, p.passos / (p.total || 1));
      $("rlBarra").style.width = `${frac * 100}%`;
      const resta = p.fps > 0 ? (p.total - p.passos) / p.fps : 0;
      const linhas = {
        "Passos": `${fmtM(p.passos)} de ${fmtM(p.total)} (${(frac * 100).toFixed(0)}%)`,
        ...(st.rodando ? { "Velocidade": `${Math.round(p.fps)} passos/s · falta ${fmtTempo(resta)}` } : {}),
        "Episodio medio": p.ep_len ? `${(p.ep_len * 0.02).toFixed(1)} s` : "-",
        "Caminhada": `${(p.vx * 100).toFixed(1)} cm/s (erro ${(p.erro_vx * 100).toFixed(1)})`,
      };
      $("rlInfo").innerHTML = Object.entries(linhas).map(([k, v]) => `<b>${k}</b><span>${v}</span>`).join("");
      historico = p.historico ?? [];
      if (!$("abaSim").hidden) desenharGrafico();
    } else {
      $("rlBarra").style.width = "0";
      $("rlInfo").innerHTML = "";
    }
    $("rlLog").textContent = (st.log ?? []).join("\n");
    if (document.activeElement !== $("rlPython")) $("rlPython").placeholder = st.python || "detectar automaticamente";
  }

  return {
    mensagem(msg) {
      if (msg.tipo === "rl_estado") mostrarTreino(msg);
      else if (msg.tipo === "rl_lista") {
        execucoes = msg.execucoes;
        desenharExecucoes();
        desenharPoliticas();
        desenharListaPoliticas();
      } else if (msg.tipo === "rl_config") {
        config = msg.config;
        padrao = msg.padrao;
        if (document.activeElement !== $("rlPassos")) $("rlPassos").value = +(config.passos / 1e6).toFixed(2);
        if (document.activeElement !== $("rlEnvs")) $("rlEnvs").value = config.envs;
        if (document.activeElement !== $("rlPython")) $("rlPython").value = config.python || "";
        desenharRecompensas();
      }
    },
    estado(mj) {
      if (!mj) {
        if ($("simLigar").checked && online()) {
          $("simInfo").innerHTML = "";
          // desligada pelo servidor (ou por outra janela); ligar a primeira vez leva alguns segundos
          if (performance.now() - ultimaTroca > 10000) $("simLigar").checked = false;
        }
        return;
      }
      sincronizar(mj);
      if (mj.ativa === "poses") {
        const linhas = {
          "Tempo": `${mj.t.toFixed(1)} s${mj.caiu ? " · CAIU (reiniciar)" : ""}`,
          "Quem move": "poses e animacoes (aba Controle)",
          "Velocidade": `${(mj.vx * 100).toFixed(1)} cm/s · giro ${(mj.giro * 180 / Math.PI).toFixed(0)} °/s`,
          "Distancia": `${(mj.dist * 100).toFixed(1)} cm`,
          "Massa": `${(mj.massa * 1000).toFixed(0)} g`,
        };
        $("simInfo").innerHTML = Object.entries(linhas).map(([k, v]) => `<b>${k}</b><span>${v}</span>`).join("");
        return;
      }
      const linhas = {
        "Tempo": `${mj.t.toFixed(1)} s${mj.caiu ? " · CAIU (reiniciar)" : ""}`,
        "Velocidade": `${(mj.vx * 100).toFixed(1)} cm/s (pedido ${(mj.cmd[0] * 100).toFixed(1)})`,
        ...(mj.omni ? {
          "Lado": `${(mj.vy * 100).toFixed(1)} cm/s (pedido ${(mj.cmd[1] * 100).toFixed(1)})`,
          "Giro": `${(mj.giro * 180 / Math.PI).toFixed(0)} °/s (pedido ${(mj.cmd[2] * 180 / Math.PI).toFixed(0)}) · rumo ${mj.rumo.toFixed(0)}°`,
        } : { "Direcao": "esta politica so anda para frente (lado/giro: v4+)" }),
        "Distancia": `${(mj.dist * 100).toFixed(1)} cm`,
        "Massa": `${(mj.massa * 1000).toFixed(0)} g`,
        "Politica": (mj.politica ? `${mj.politica} (${fmtM(mj.passos_politica)})` : "nenhuma") +
          (mj.politica_giro ? ` · giro: ${mj.politica_giro}` : ""),
        "Ativa": mj.ativa === "giro" ? "politica de giro" : "caminhada",
      };
      $("simInfo").innerHTML = Object.entries(linhas).map(([k, v]) => `<b>${k}</b><span>${v}</span>`).join("");
    },
    seguirCamera: () => $("simSeguir").checked,
  };
}
