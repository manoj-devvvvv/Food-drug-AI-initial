const $ = (sel) => document.querySelector(sel);
const state = { mode: "S", last: null, loading: false, controller: null };

const MODE_LABEL = { S: "Specific labels", P: "Paper-like labels" };
const BADGE = {
  ok: (d) => ({ 0: "Avoid together", 1: "Take with food", 2: "Non-significant" }[d.predicted_class] || "Prediction"),
  rule: () => "Rule layer",
  evidence: () => "Dataset evidence",
  uncertain: () => "Low confidence · abstained",
  not_supported: () => "Not supported",
};

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

/* ---------- health ---------- */
async function loadHealth() {
  const pill = $("#health");
  try {
    const res = await fetch("/api/health");
    if (!res.ok) throw new Error("health");
    const data = await res.json();
    pill.className = "pill pill-ok";
    pill.replaceChildren(el("i"), el("span", null,
      `Ready · ${data.drugs.toLocaleString()} drugs · ${data.compounds.toLocaleString()} compounds`));
  } catch {
    pill.className = "pill pill-fail";
    pill.replaceChildren(el("i"), el("span", null, "API offline"));
  }
}

/* ---------- examples ---------- */
async function loadExamples() {
  try {
    const res = await fetch("/api/examples");
    const { pairs } = await res.json();
    const box = $("#chips");
    pairs.forEach((p) => {
      const shortFood = p.food.length > 22 ? `${p.food.slice(0, 20)}…` : p.food;
      const chip = el("button", "chip", `${p.drug} + ${shortFood}`);
      chip.type = "button";
      chip.title = `${p.drug} + ${p.food}`;
      chip.append(el("small", null, p.note));
      chip.addEventListener("click", () => {
        $("#drug").value = p.drug;
        $("#food").value = p.food;
        runPrediction();
      });
      box.append(chip);
    });
  } catch {
    /* chips are optional */
  }
}

/* ---------- suggestions ---------- */
function wireSuggestions(inputId, listId, kind) {
  const input = $(`#${inputId}`);
  const list = $(`#${listId}`);
  let timer = null;
  let controller = null;

  input.addEventListener("input", () => {
    clearTimeout(timer);
    const q = input.value.trim();
    if (q.length < 1) {
      list.replaceChildren();
      return;
    }
    timer = setTimeout(async () => {
      controller?.abort();
      controller = new AbortController();
      try {
        const res = await fetch(`/api/suggest?kind=${kind}&q=${encodeURIComponent(q)}&limit=8`,
          { signal: controller.signal });
        if (!res.ok) return;
        const { suggestions } = await res.json();
        list.replaceChildren(...suggestions.map((s) => {
          const opt = document.createElement("option");
          opt.value = s;
          return opt;
        }));
      } catch (err) {
        if (err.name !== "AbortError") console.warn(err);
      }
    }, 150);
  });
}

/* ---------- rendering ---------- */
function highlight(parent, text, terms) {
  const hits = terms
    .filter(Boolean)
    .map((t) => ({ t, i: text.toLowerCase().indexOf(t.toLowerCase()) }))
    .filter((h) => h.i >= 0)
    .sort((a, b) => a.i - b.i);
  let cursor = 0;
  hits.forEach(({ t, i }) => {
    if (i < cursor) return;
    if (i > cursor) parent.append(document.createTextNode(text.slice(cursor, i)));
    parent.append(el("span", "em", text.slice(i, i + t.length)));
    cursor = i + t.length;
  });
  if (cursor < text.length) parent.append(document.createTextNode(text.slice(cursor)));
}

function severityFor(data) {
  if (data.status === "uncertain") return "warn";
  if (data.status === "rule") return "rule";
  if (data.status === "not_supported") return "unsupported";
  return data.severity || "neutral";
}

function renderError(message) {
  const root = $("#result");
  root.replaceChildren();
  const box = el("div", "error-box");
  box.append(el("h3", null, "Could not run the prediction"));
  box.append(el("p", null, message));
  root.append(box);
}

function renderUnsupported(data) {
  const root = $("#result");
  const sev = "unsupported";
  const verdict = el("div", "verdict");
  verdict.dataset.sev = sev;
  const top = el("div", "verdict-top");
  top.append(el("span", "badge", BADGE.not_supported()));
  top.dataset.sev = sev;
  verdict.append(top);
  if (data.drug || data.inputs) {
    const pair = [data.drug || data.inputs?.drug, data.food || data.inputs?.food].filter(Boolean).join(" + ");
    if (pair) verdict.append(el("p", "sentence", pair));
  }
  verdict.append(el("p", "reason", data.reason || "This input cannot be scored by the model."));
  root.replaceChildren(verdict);
  root.append(disclaimerStrip(data.disclaimer));
}

function disclaimerStrip(text) {
  const strip = el("div", "disclaimer-strip");
  strip.innerHTML = `<svg viewBox="0 0 20 20" width="15" height="15" aria-hidden="true"><path d="M10 1.8 18.2 16H1.8L10 1.8Zm0 4.6a.9.9 0 0 0-.9.9v3.4a.9.9 0 0 0 1.8 0V7.3a.9.9 0 0 0-.9-.9Zm0 6.2a.95.95 0 1 0 0 1.9.95.95 0 0 0 0-1.9Z" fill="currentColor"/></svg>`;
  strip.append(el("span", null, text || "Research tool; confirm with a physician or pharmacist."));
  return strip;
}

function renderResult(data) {
  const root = $("#result");
  root.replaceChildren();
  if (data.status === "not_supported") return renderUnsupported(data);

  const sev = severityFor(data);
  const verdict = el("div", "verdict");
  verdict.dataset.sev = sev;

  const top = el("div", "verdict-top");
  const badge = el("span", "badge", BADGE[data.status] ? BADGE[data.status](data) : "Prediction");
  badge.dataset.sev = sev;
  top.append(badge);
  const bits = [MODE_LABEL[data.mode] || data.mode];
  if (typeof data.confidence === "number") bits.push(`confidence ${(data.confidence * 100).toFixed(1)}%`);
  bits.push(`${data.latency_ms?.toFixed?.(0) ?? "—"} ms`);
  top.append(el("span", "meta-inline", bits.join(" · ")));
  verdict.append(top);

  const sentence = el("p", "sentence");
  highlight(sentence, data.sentence || "", [data.drug, data.food]);
  verdict.append(sentence);

  if (data.reason) verdict.append(el("p", "reason", data.reason));
  root.append(verdict);

  if (Array.isArray(data.probability_rows) && data.probability_rows.length) {
    const best = data.probability_rows.reduce((a, b) => (b.probability > a.probability ? b : a));
    const probs = el("div", "probs");
    data.probability_rows.forEach((row, i) => {
      const line = el("div", "prob");
      line.dataset.class = String(row.class);
      if (row.class === best.class) line.classList.add("best");
      line.append(el("span", "prob-label", row.label));
      const track = el("div", "prob-track");
      const fill = el("div", "prob-fill");
      track.append(fill);
      line.append(track, el("span", "prob-value", `${(row.probability * 100).toFixed(1)}%`));
      probs.append(line);
      setTimeout(() => {
        fill.style.width = `${Math.max(row.probability * 100, 1.5)}%`;
      }, 40 + i * 70);
    });
    root.append(probs);
  }

  const details = el("dl", "details");
  const addDetail = (label, value, mono = false) => {
    if (value === undefined || value === null || value === "") return;
    const wrap = el("div", "detail");
    wrap.append(el("dt", null, label));
    wrap.append(el("dd", mono ? "mono" : null, value));
    details.append(wrap);
  };
  addDetail("Matched drug", data.drug);
  addDetail("Matched compound", data.food);
  addDetail("Applicability domain", typeof data.domain_similarity === "number"
    ? `${(data.domain_similarity * 100).toFixed(0)}% nearest training drug` : null);
  addDetail("Label mode", MODE_LABEL[data.mode] || data.mode);
  addDetail("Input drug", data.inputs?.drug, true);
  addDetail("Input food", data.inputs?.food, true);
  addDetail("Match note", data.note);
  if (data.extrapolation) {
    addDetail("Domain warning", "Drug is outside this mode's training labels; treat this as an extrapolation.");
  }
  (data.candidates || []).forEach((c, i) => {
    const cls = { 0: "avoid", 1: "beneficial", 2: "non-significant" }[c.class] ?? `class ${c.class}`;
    const conf = typeof c.confidence === "number" ? ` · ${(c.confidence * 100).toFixed(0)}%` : "";
    addDetail(`Candidate ${i + 1}`, `${c.compound} · ${cls}${conf}`);
  });
  if (details.childElementCount) root.append(details);

  root.append(disclaimerStrip(data.disclaimer));
}

function showLoading() {
  const root = $("#result");
  root.setAttribute("aria-busy", "true");
  root.replaceChildren();
  const skel = el("div", "skeleton");
  ["w40 tall", "w90", "w70", "w90", "w70"].forEach((c) => {
    const bar = el("div", `skel ${c}`);
    skel.append(bar);
  });
  root.append(skel);
}

/* ---------- prediction ---------- */
async function runPrediction() {
  const drug = $("#drug").value.trim();
  const food = $("#food").value.trim();
  const error = $("#form-error");
  error.textContent = "";
  if (!drug || !food) {
    error.textContent = "Enter both a drug and a food compound.";
    (!drug ? $("#drug") : $("#food")).focus();
    return;
  }

  state.controller?.abort();
  const controller = new AbortController();
  state.controller = controller;
  state.loading = true;
  const btn = $("#submit");
  btn.classList.add("loading");
  btn.disabled = true;
  btn.querySelector(".btn-label").textContent = "Scoring…";
  showLoading();

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ drug, food, mode: state.mode }),
      signal: controller.signal,
    });
    const data = await res.json();
    if (!res.ok) {
      renderError(data.detail || `Request failed with status ${res.status}.`);
    } else {
      state.last = { drug, food };
      renderResult(data);
    }
  } catch (err) {
    if (err.name === "AbortError") return;
    renderError("The API is unreachable. Start it with: uvicorn web.app:app --port 8000");
  } finally {
    if (state.controller === controller) {
      state.controller = null;
      state.loading = false;
      btn.classList.remove("loading");
      btn.disabled = false;
      btn.querySelector(".btn-label").textContent = "Predict interaction";
      $("#result").removeAttribute("aria-busy");
    }
  }
}

/* ---------- wiring ---------- */
$("#form").addEventListener("submit", (event) => {
  event.preventDefault();
  runPrediction();
});

$("#swap").addEventListener("click", () => {
  const drug = $("#drug");
  const food = $("#food");
  [drug.value, food.value] = [food.value, drug.value];
  drug.focus();
});

document.querySelectorAll(".segmented button").forEach((btn) => {
  btn.addEventListener("click", () => {
    if (btn.dataset.mode === state.mode) return;
    state.mode = btn.dataset.mode;
    document.querySelectorAll(".segmented button").forEach((b) => {
      const on = b.dataset.mode === state.mode;
      b.classList.toggle("active", on);
      b.setAttribute("aria-pressed", String(on));
    });
    if (state.last) runPrediction();
  });
});

wireSuggestions("drug", "drug-list", "drug");
wireSuggestions("food", "food-list", "food");
loadHealth();
loadExamples();
