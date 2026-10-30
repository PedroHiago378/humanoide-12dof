// Poses salvas, transicoes e editor de animacao por keyframe.
// A interpolacao espelha controle/animacao.py (mantenha os dois iguais): o servidor toca a
// animacao (e manda aos servos); aqui ela e usada para desenhar as curvas e para o modo offline.

export const TIPOS = {
  linear: "linear",
  suave: "suave (cosseno)",
  min_jerk: "jerk minimo",
  acelerar: "acelerar",
  desacelerar: "desacelerar",
  degrau: "degrau (segura e pula)",
  spline: "spline (passa suave pelos keyframes)",
};

export function forma(tipo, s) {
  s = Math.min(1, Math.max(0, s));
  switch (tipo) {
    case "linear": return s;
    case "min_jerk": return s ** 3 * (10 - 15 * s + 6 * s * s);
    case "acelerar": return s * s;
    case "desacelerar": return 1 - (1 - s) ** 2;
    case "degrau": return s >= 1 ? 1 : 0;
    default: return (1 - Math.cos(Math.PI * s)) / 2; // suave
  }
}

export const duracao = (anim) => Math.max(0, ...(anim.keyframes ?? []).map((k) => k.t));

// pose (graus) no tempo t, na ordem de `nomes`
export function avaliar(anim, t, nomes) {
  const ks = [...(anim.keyframes ?? [])].sort((a, b) => a.t - b.t);
  if (!ks.length) return nomes.map(() => 0);
  const P = (k, n) => ks[k].q[n] ?? 0;
  if (ks.length === 1 || t <= ks[0].t) return nomes.map((n) => P(0, n));
  if (t >= ks.at(-1).t) return nomes.map((n) => P(ks.length - 1, n));
  let i = ks.findIndex((k) => k.t > t);            // trecho ks[i-1] -> ks[i]
  const t0 = ks[i - 1].t, t1 = ks[i].t, h = t1 - t0;
  const s = h > 0 ? (t - t0) / h : 1;
  const padrao = ks[i].interp || anim.interp || "suave";
  return nomes.map((n) => {
    const tipo = anim.interp_juntas?.[n] || padrao;
    const p0 = P(i - 1, n), p1 = P(i, n);
    if (tipo !== "spline") return p0 + (p1 - p0) * forma(tipo, s);
    const m0 = i >= 2 && t1 > ks[i - 2].t ? (p1 - P(i - 2, n)) / (t1 - ks[i - 2].t) : 0;
    const m1 = i + 1 < ks.length && ks[i + 1].t > t0 ? (P(i + 1, n) - p0) / (ks[i + 1].t - t0) : 0;
    const s2 = s * s, s3 = s2 * s;
    return (2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * h * m0 + (-2 * s3 + 3 * s2) * p1 + (s3 - s2) * h * m1;
  });
}

const novaAnim = () => ({ interp: "suave", interp_juntas: {}, keyframes: [] });
const copia = (x) => JSON.parse(JSON.stringify(x));
const opcoes = (padrao) => (padrao ? `<option value="">${padrao}</option>` : "") +
  Object.entries(TIPOS).map(([k, v]) => `<option value="${k}">${v}</option>`).join("");

/**
 * ctx: { nomes, limites(nome) -> [lo, hi] graus, velMax(nome) -> graus/s, getQ() -> {junta: graus},
 *        setQ({junta: graus}), enviar(msg), online(), posesFabrica: {nome: {junta: graus}}, aviso(texto) }
 */
export function iniciarAnimacao(ctx) {
  const $ = (id) => document.getElementById(id);
  const css = getComputedStyle(document.documentElement);
  const cor = (v) => css.getPropertyValue(v).trim();

  let biblioteca = { poses: Object.fromEntries(Object.entries(ctx.posesFabrica).map(([n, q]) => [n, { q, fabrica: true }])), animacoes: {} };
  let anim = novaAnim();
  let nomeAtual = "";
  let sujo = false;
  let sel = -1;            // keyframe selecionado
  let tCursor = 0;
  let tocando = false;     // remoto (servidor) ou local
  let local = null;        // reproducao/transicao local (offline): funcao chamada a cada quadro
  let redesenhar = true;

  // ------------------------------------------------------------- poses
  $("poseInterp").innerHTML = opcoes();
  $("poseInterp").value = "min_jerk";

  function desenharPoses() {
    $("poses").innerHTML = "";
    for (const [nome, p] of Object.entries(biblioteca.poses)) {
      const chip = document.createElement("span");
      chip.className = "chip" + (p.fabrica ? "" : " usuario");
      chip.innerHTML = `<button class="ir"></button>${p.fabrica ? "" : `<button class="x" title="apagar pose">&times;</button>`}`;
      chip.querySelector(".ir").textContent = nome.replace(/_/g, " ");
      chip.querySelector(".ir").onclick = () => irPara(p.q, +$("poseDur").value || 0, $("poseInterp").value);
      const x = chip.querySelector(".x");
      if (x) x.onclick = () => { if (confirm(`Apagar a pose "${nome}"?`)) ctx.enviar({ tipo: "apagar_pose", nome }); };
      $("poses").appendChild(chip);
    }
  }

  function poseCompleta(q) {
    return Object.fromEntries(ctx.nomes.map((n) => [n, q[n] ?? 0]));
  }

  function irPara(q, dur, interp) {
    q = poseCompleta(q);
    if (ctx.online()) return ctx.enviar({ tipo: "pose", q, duracao: dur, interp });
    const ini = ctx.getQ();
    const t0 = performance.now();
    local = () => {
      const s = dur > 0 ? (performance.now() - t0) / (dur * 1000) : 1;
      const f = forma(interp === "spline" ? "suave" : interp, s);
      ctx.setQ(Object.fromEntries(ctx.nomes.map((n) => [n, ini[n] + (q[n] - ini[n]) * f])));
      if (s >= 1) local = null;
    };
  }

  $("poseSalvar").onclick = () => {
    const nome = $("poseNome").value.trim();
    if (!nome) return ctx.aviso("De um nome para a pose.");
    if (!ctx.online()) return ctx.aviso("Salvar precisa do servidor (abra pelo programa).");
    if (biblioteca.poses[nome] && !confirm(`Substituir a pose "${nome}"?`)) return;
    ctx.enviar({ tipo: "salvar_pose", nome, valor: ctx.getQ() });
    $("poseNome").value = "";
  };

  // ------------------------------------------------------------- lista de animacoes
  function desenharLista() {
    const nomes = Object.keys(biblioteca.animacoes).sort();
    $("animLista").innerHTML = `<option value="">(animacoes salvas)</option>` +
      nomes.map((n) => `<option>${n.replace(/</g, "&lt;")}</option>`).join("");
    $("animLista").value = nomes.includes(nomeAtual) ? nomeAtual : "";
  }

  function carregar(a, nome) {
    anim = { ...novaAnim(), ...copia(a) };
    anim.keyframes.sort((x, y) => x.t - y.t);
    nomeAtual = nome;
    $("animNome").value = nome;
    sel = anim.keyframes.length ? 0 : -1;
    tCursor = 0;
    marcar(false);
    atualizarCampos();
  }

  function marcar(s = true) {
    sujo = s;
    $("animNome").classList.toggle("sujo", sujo);
    redesenhar = true;
  }

  const descartarAlteracoes = () => !sujo || confirm("Descartar as alteracoes nao salvas desta animacao?");

  $("animLista").onchange = () => {
    const nome = $("animLista").value;
    if (!nome || !descartarAlteracoes()) return desenharLista();
    carregar(biblioteca.animacoes[nome], nome);
  };
  $("animNova").onclick = () => { if (descartarAlteracoes()) carregar(novaAnim(), ""); desenharLista(); };
  $("animSalvar").onclick = () => {
    const nome = $("animNome").value.trim();
    if (!nome) return ctx.aviso("De um nome para a animacao.");
    if (!ctx.online()) return ctx.aviso("Salvar precisa do servidor (abra pelo programa).");
    if (nome !== nomeAtual && biblioteca.animacoes[nome] && !confirm(`Substituir a animacao "${nome}"?`)) return;
    ctx.enviar({ tipo: "salvar_anim", nome, valor: anim });
    nomeAtual = nome;
    marcar(false);
  };
  $("animApagar").onclick = () => {
    if (!nomeAtual || !biblioteca.animacoes[nomeAtual]) return ctx.aviso("Essa animacao ainda nao foi salva.");
    if (confirm(`Apagar a animacao "${nomeAtual}"?`)) ctx.enviar({ tipo: "apagar_anim", nome: nomeAtual });
  };
  $("animExportar").onclick = () => {
    const blob = new Blob([JSON.stringify({ nome: $("animNome").value || "animacao", ...anim }, null, 1)], { type: "application/json" });
    const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: `${$("animNome").value || "animacao"}.json` });
    a.click();
    URL.revokeObjectURL(a.href);
  };
  $("animImportar").onclick = () => $("animArquivo").click();
  $("animArquivo").onchange = async (e) => {
    const f = e.target.files[0];
    e.target.value = "";
    if (!f || !descartarAlteracoes()) return;
    try {
      const a = JSON.parse(await f.text());
      if (!Array.isArray(a.keyframes)) throw new Error("arquivo sem keyframes");
      carregar(a, a.nome ?? f.name.replace(/\.json$/i, ""));
      marcar(true);
    } catch (err) { ctx.aviso(`Nao deu para importar: ${err.message}`); }
  };

  // ------------------------------------------------------------- keyframes
  $("kfInterp").innerHTML = opcoes("padrao da animacao");
  $("animInterp").innerHTML = opcoes();
  $("curvaInterp").innerHTML = opcoes("padrao (keyframe)");
  $("curvaJunta").innerHTML = ctx.nomes.map((n) => `<option>${n}</option>`).join("");

  function atualizarCampos() {
    const k = anim.keyframes[sel];
    for (const id of ["kfT", "kfInterp", "kfAtualizar", "kfApagar"]) $(id).disabled = !k;
    $("kfT").value = k ? k.t.toFixed(2) : "";
    $("kfInterp").value = k?.interp ?? "";
    $("animInterp").value = anim.interp ?? "suave";
    $("curvaInterp").value = anim.interp_juntas?.[$("curvaJunta").value] ?? "";
    redesenhar = true;
  }

  function ordenar() {
    const k = anim.keyframes[sel];
    anim.keyframes.sort((a, b) => a.t - b.t);
    sel = anim.keyframes.indexOf(k);
  }

  $("kfAdd").onclick = () => {
    const t = anim.keyframes.length ? Math.round(tCursor * 100) / 100 : 0;
    const existente = anim.keyframes.findIndex((k) => Math.abs(k.t - t) < 0.01);
    if (existente >= 0) {
      anim.keyframes[existente].q = ctx.getQ();
      sel = existente;
    } else {
      anim.keyframes.push({ t, q: ctx.getQ() });
      sel = anim.keyframes.length - 1;
      ordenar();
    }
    tCursor = t + 1;   // proximo keyframe 1 s depois
    marcar();
    atualizarCampos();
  };
  $("kfAtualizar").onclick = () => { anim.keyframes[sel].q = ctx.getQ(); marcar(); };
  $("kfApagar").onclick = () => {
    anim.keyframes.splice(sel, 1);
    sel = Math.min(sel, anim.keyframes.length - 1);
    marcar();
    atualizarCampos();
  };
  $("kfT").onchange = () => {
    anim.keyframes[sel].t = Math.max(0, Math.round(+$("kfT").value * 100) / 100 || 0);
    ordenar();
    marcar();
    atualizarCampos();
  };
  $("kfInterp").onchange = () => {
    const k = anim.keyframes[sel];
    if ($("kfInterp").value) k.interp = $("kfInterp").value; else delete k.interp;
    marcar();
  };
  $("animInterp").onchange = () => { anim.interp = $("animInterp").value; marcar(); };
  $("curvaJunta").onchange = atualizarCampos;
  $("curvaInterp").onchange = () => {
    anim.interp_juntas ??= {};
    const n = $("curvaJunta").value;
    if ($("curvaInterp").value) anim.interp_juntas[n] = $("curvaInterp").value;
    else delete anim.interp_juntas[n];
    marcar();
  };

  // ------------------------------------------------------------- tocar / cursor
  let ultimoIr = 0;
  function irTempo(t, forcar = false) {
    tCursor = Math.max(0, t);
    redesenhar = true;
    if (!anim.keyframes.length) return;
    if (ctx.online()) {
      const agora = performance.now();
      if (!forcar && agora - ultimoIr < 60) return;   // ~16 Hz enquanto arrasta
      ultimoIr = agora;
      ctx.enviar({ tipo: "anim_ir", anim, t: tCursor });
    } else {
      local = null;
      const v = avaliar(anim, tCursor, ctx.nomes);
      ctx.setQ(Object.fromEntries(ctx.nomes.map((n, i) => [n, v[i]])));
    }
  }

  function tocar() {
    if (anim.keyframes.length < 2) return ctx.aviso("A animacao precisa de pelo menos 2 keyframes.");
    const dur = duracao(anim);
    const inicio = tCursor >= dur - 1e-3 ? 0 : tCursor;
    const loop = $("animLoop").checked;
    tocando = true;
    if (ctx.online()) return ctx.enviar({ tipo: "anim_tocar", anim, inicio, loop });
    const t0 = performance.now() - inicio * 1000;
    local = () => {
      let t = (performance.now() - t0) / 1000;
      if (loop) t %= dur;
      else if (t >= dur) { t = dur; tocando = false; local = null; }
      tCursor = t;
      const v = avaliar(anim, t, ctx.nomes);
      ctx.setQ(Object.fromEntries(ctx.nomes.map((n, i) => [n, v[i]])));
      redesenhar = true;
    };
  }

  function parar() {
    tocando = false;
    local = null;
    if (ctx.online()) ctx.enviar({ tipo: "anim_parar" });
    redesenhar = true;
  }

  $("animTocar").onclick = () => (tocando ? parar() : tocar());
  $("animParar").onclick = () => { parar(); irTempo(0, true); };
  window.addEventListener("keydown", (e) => {
    if (e.code !== "Space" || /INPUT|SELECT|TEXTAREA|BUTTON/.test(document.activeElement?.tagName)) return;
    e.preventDefault();
    tocando ? parar() : tocar();
  });

  // ------------------------------------------------------------- linha do tempo (canvas)
  const cv = $("ltCanvas");
  const PAD = 14;
  const vistaDur = () => Math.max(2, duracao(anim) * 1.1 + 0.3);
  const xDe = (t, w) => PAD + (t / vistaDur()) * (w - 2 * PAD);
  const tDe = (x, w) => ((x - PAD) / (w - 2 * PAD)) * vistaDur();

  function ajustar(c) {
    const r = c.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    if (c.width !== Math.round(r.width * dpr) || c.height !== Math.round(r.height * dpr)) {
      c.width = Math.round(r.width * dpr);
      c.height = Math.round(r.height * dpr);
    }
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    return [g, r.width, r.height];
  }

  function passoRegua() {
    const v = vistaDur();
    return v <= 3 ? 0.25 : v <= 8 ? 0.5 : v <= 20 ? 1 : 5;
  }

  function desenharLinha() {
    const [g, w, h] = ajustar(cv);
    g.clearRect(0, 0, w, h);
    const passo = passoRegua();
    g.font = "10px system-ui";
    g.textAlign = "center";
    for (let t = 0; t <= vistaDur() + 1e-6; t += passo) {
      const x = xDe(t, w);
      const inteiro = Math.abs(t - Math.round(t)) < 1e-6;
      g.strokeStyle = inteiro ? cor("--borda") : "#22272f";
      g.beginPath(); g.moveTo(x, 14); g.lineTo(x, h); g.stroke();
      if (inteiro || passo >= 1) { g.fillStyle = cor("--fraco"); g.fillText(`${+t.toFixed(2)}s`, x, 10); }
    }
    // fim da animacao
    if (anim.keyframes.length) {
      const xf = xDe(duracao(anim), w);
      g.fillStyle = "rgba(79,156,249,.08)";
      g.fillRect(xDe(0, w), 16, xf - xDe(0, w), h - 18);
    }
    const ym = (h + 14) / 2;
    anim.keyframes.forEach((k, i) => {
      const x = xDe(k.t, w);
      g.save();
      g.translate(x, ym);
      g.rotate(Math.PI / 4);
      g.fillStyle = i === sel ? "#ffb000" : cor("--acento");
      g.strokeStyle = "#0b0d10";
      g.fillRect(-5, -5, 10, 10);
      g.strokeRect(-5, -5, 10, 10);
      g.restore();
      if (k.interp) { g.fillStyle = cor("--fraco"); g.fillText(k.interp, x, ym + 18); }
    });
    const xc = xDe(tCursor, w);
    g.strokeStyle = "#ff3366";
    g.lineWidth = 1.5;
    g.beginPath(); g.moveTo(xc, 12); g.lineTo(xc, h); g.stroke();
    g.lineWidth = 1;
    $("animTempo").textContent = `${tCursor.toFixed(2)} / ${duracao(anim).toFixed(2)} s`;
    $("animTocar").innerHTML = tocando ? "&#10074;&#10074;" : "&#9654;";
    $("animTocar").title = tocando ? "pausar (espaco)" : "tocar (espaco)";
  }

  let arrasto = null;
  cv.addEventListener("pointerdown", (e) => {
    const r = cv.getBoundingClientRect();
    const x = e.clientX - r.left;
    const i = anim.keyframes.findIndex((k) => Math.abs(xDe(k.t, r.width) - x) < 8);
    cv.setPointerCapture(e.pointerId);
    if (i >= 0) {
      sel = i;
      arrasto = { tipo: "kf", moveu: false };
      atualizarCampos();
      if (tocando) parar();
      irTempo(anim.keyframes[i].t, true);
    } else {
      if (tocando) parar();
      arrasto = { tipo: "cursor" };
      irTempo(tDe(x, r.width));
    }
  });
  cv.addEventListener("pointermove", (e) => {
    if (!arrasto) return;
    const r = cv.getBoundingClientRect();
    let t = Math.max(0, tDe(e.clientX - r.left, r.width));
    if (arrasto.tipo === "kf") {
      if (!e.shiftKey) t = Math.round(t / 0.05) * 0.05;   // shift = sem encaixe
      anim.keyframes[sel].t = Math.round(t * 1000) / 1000;
      arrasto.moveu = true;
      tCursor = anim.keyframes[sel].t;
      atualizarCampos();
    } else irTempo(t);
  });
  cv.addEventListener("pointerup", () => {
    if (arrasto?.tipo === "kf" && arrasto.moveu) { ordenar(); marcar(); atualizarCampos(); }
    else if (arrasto?.tipo === "cursor") irTempo(tCursor, true);
    arrasto = null;
  });

  // ------------------------------------------------------------- curva da junta
  const cc = $("curvaCanvas");
  function desenharCurva() {
    const [g, w, h] = ajustar(cc);
    g.clearRect(0, 0, w, h);
    const n = $("curvaJunta").value;
    const [lo, hi] = ctx.limites(n);
    const yDe = (v) => 6 + (1 - (v - lo) / (hi - lo || 1)) * (h - 12);
    g.strokeStyle = "#22272f";
    for (let t = 0; t <= vistaDur(); t += passoRegua()) {
      const x = xDe(t, w);
      g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke();
    }
    g.strokeStyle = cor("--borda");
    g.setLineDash([3, 3]);
    g.beginPath(); g.moveTo(PAD, yDe(0)); g.lineTo(w - PAD, yDe(0)); g.stroke();
    g.setLineDash([]);
    g.fillStyle = cor("--fraco");
    g.font = "10px system-ui";
    g.textAlign = "left";
    g.fillText(`${hi.toFixed(0)}°`, 2, 12);
    g.fillText(`${lo.toFixed(0)}°`, 2, h - 3);
    if (!anim.keyframes.length) return ($("curvaInfo").textContent = "");
    const dur = duracao(anim);
    const N = 400;
    let vmax = 0, ant = null;
    g.strokeStyle = cor("--acento");
    g.lineWidth = 2;
    g.beginPath();
    for (let k = 0; k <= N; k++) {
      const t = (k / N) * vistaDur();
      const v = avaliar(anim, t, [n])[0];
      if (ant !== null && t <= dur) vmax = Math.max(vmax, Math.abs(v - ant) / (vistaDur() / N));
      ant = v;
      const x = xDe(t, w), y = yDe(v);
      k ? g.lineTo(x, y) : g.moveTo(x, y);
    }
    g.stroke();
    g.lineWidth = 1;
    anim.keyframes.forEach((k, i) => {
      g.fillStyle = i === sel ? "#ffb000" : "#e3e6eb";
      g.beginPath(); g.arc(xDe(k.t, w), yDe(k.q[n] ?? 0), 3.5, 0, 2 * Math.PI); g.fill();
    });
    const xc = xDe(tCursor, w);
    g.strokeStyle = "#ff3366";
    g.beginPath(); g.moveTo(xc, 0); g.lineTo(xc, h); g.stroke();
    const lim = ctx.velMax(n);
    $("curvaInfo").textContent = `vel. max ${vmax.toFixed(0)}°/s` + (vmax > lim ? ` (servo: ${lim.toFixed(0)}°/s!)` : "");
    $("curvaInfo").style.color = vmax > lim ? cor("--erro") : "";
  }

  // ------------------------------------------------------------- API para o app
  carregar(novaAnim(), "");
  desenharPoses();
  desenharLista();
  new ResizeObserver(() => { redesenhar = true; }).observe($("linhaTempo"));

  return {
    conectou() { ctx.enviar({ tipo: "biblioteca" }); },
    biblioteca(msg) {
      biblioteca = { poses: msg.poses, animacoes: msg.animacoes };
      desenharPoses();
      desenharLista();
    },
    // estado vindo do servidor (msg.anim do controlador de animacao)
    estado(animUi, controlador) {
      const tocandoAntes = tocando;
      if (controlador === "animacao" && animUi) {
        tocando = animUi.tocando;
        if (tocando || tocandoAntes) tCursor = animUi.t;
      } else if (ctx.online()) tocando = false;
      if (tocando || tocando !== tocandoAntes) redesenhar = true;
    },
    pararLocal() { local = null; },
    selecionarJunta(n) { $("curvaJunta").value = n; atualizarCampos(); },
    quadro() {
      if (local) local();
      if (redesenhar && !document.body.classList.contains("lt-fechada")) {
        redesenhar = false;
        desenharLinha();
        desenharCurva();
      }
    },
  };
}
