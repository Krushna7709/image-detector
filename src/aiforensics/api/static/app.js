const $ = (id) => document.getElementById(id);
const GAUGE_CIRCUMFERENCE = 515.22;
const MAX_BYTES = 15 * 1024 * 1024;
const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"];
const STRESS_LEVELS = [
  ["none", "Original"],
  ["jpeg50", "JPEG q50"],
  ["jpeg30", "JPEG q30"],
  ["resize50", "Downscaled 50%"],
  ["blur2", "Blur σ2"],
];
const LOADING_STEPS = [
  "Hashing file…",
  "Reading metadata & provenance…",
  "Measuring noise field…",
  "Fitting spectral decay…",
  "Localizing inconsistencies…",
  "Fusing evidence…",
  "Rendering explainability maps…",
];

const verdictCard = $("verdictCard");
const gaugeValue = $("gaugeValue");
const gaugeNumber = $("gaugeNumber");
const verdictChip = $("verdictChip");
const verdictTitle = $("verdictTitle");
const verdictSub = $("verdictSub");
const signalList = $("signalList");
const exampleGrid = $("exampleGrid");
const dropzone = $("dropzone");
const fileInput = $("fileInput");
const dzPreview = $("dzPreview");
const fileRow = $("fileRow");
const fileName = $("fileName");
const fileSize = $("fileSize");
const intakeResult = $("intakeResult");
const intakeResultValue = $("intakeResultValue");
const intakeResultMeta = $("intakeResultMeta");
const distortionSelect = $("distortionSelect");
const analyzeBtn = $("analyzeBtn");
const loadingCard = $("loadingCard");
const loadingText = $("loadingText");
const results = $("results");
const howGrid = $("howGrid");
const heatImage = $("heatImage");
const boxImage = $("boxImage");
const toggleHeat = $("toggleHeat");
const toggleBoxes = $("toggleBoxes");
const heatOpacity = $("heatOpacity");
const toastEl = $("toast");

let selectedFile = null;
let currentAnalysis = null;
let busy = false;
let toastTimer = null;
let loadingTimer = null;
let previewUrl = null;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function pct(value, digits = 0) {
  return `${(value * 100).toFixed(digits)}%`;
}

function humanSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes;
  let index = -1;
  do {
    value /= 1024;
    index += 1;
  } while (value >= 1024 && index < units.length - 1);
  return `${value.toFixed(value >= 10 ? 0 : 1)} ${units[index]}`;
}

function showToast(message) {
  toastEl.textContent = message;
  toastEl.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.add("hidden"), 5200);
}

async function requestJson(url, options) {
  let response;
  try {
    response = await fetch(url, options);
  } catch {
    throw new Error("Could not reach the Image Detector server. Is it running?");
  }
  let body = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  if (!response.ok) {
    let message = body && (body.message || body.error || body.detail);
    if (Array.isArray(message)) message = message.map((item) => item.msg || "invalid input").join("; ");
    throw new Error(typeof message === "string" && message ? message : `Request failed (${response.status})`);
  }
  return body;
}

function setBusy(value) {
  analyzeBtn.disabled = value;
  analyzeBtn.querySelector("span").textContent = value ? "Analyzing…" : "Analyze image";
  const stressBtn = $("stressBtn");
  if (stressBtn) stressBtn.disabled = value || !currentAnalysis;
}

function startLoading() {
  loadingCard.classList.remove("hidden");
  howGrid.classList.add("hidden");
  results.classList.add("hidden");
  hideIntakeResult();
  loadingCard.scrollIntoView({ behavior: "smooth", block: "center" });
  let index = 0;
  loadingText.textContent = LOADING_STEPS[0];
  clearInterval(loadingTimer);
  loadingTimer = setInterval(() => {
    index = (index + 1) % LOADING_STEPS.length;
    loadingText.textContent = LOADING_STEPS[index];
  }, 650);
}

function stopLoading() {
  clearInterval(loadingTimer);
  loadingTimer = null;
  loadingCard.classList.add("hidden");
}

async function analyzeSource(source, distortion) {
  const form = new FormData();
  form.append("distortion", distortion);
  if (source.kind === "file") {
    form.append("file", source.file, source.file.name);
    return requestJson("/api/analyze", { method: "POST", body: form });
  }
  return requestJson(`/api/analyze/example/${encodeURIComponent(source.name)}`, {
    method: "POST",
    body: form,
  });
}

async function runAnalysis(source) {
  if (busy) return;
  busy = true;
  setBusy(true);
  startLoading();
  try {
    const result = await analyzeSource(source, distortionSelect.value);
    currentAnalysis = { source, result, distortion: distortionSelect.value };
    renderResult(result);
    results.classList.remove("hidden");
    results.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showToast(error.message);
    if (currentAnalysis) results.classList.remove("hidden");
    else howGrid.classList.remove("hidden");
  } finally {
    stopLoading();
    busy = false;
    setBusy(false);
  }
}

function renderResult(result) {
  renderVerdict(result);
  renderSignals(result.signals);
  renderViewer(result);
  renderAttribution(result);
  renderExplain(result.explain);
  renderMetadata(result.metadata);
  renderNotes(result);
  renderReport(result);
  renderIntakeResult(result);
  resetStress();
}

function renderVerdict(result) {
  const { verdict } = result;
  verdictCard.classList.remove("is-ai", "is-real", "is-uncertain");
  verdictCard.classList.add(
    verdict.label === "ai_generated" ? "is-ai" : verdict.label === "real" ? "is-real" : "is-uncertain"
  );

  const copy = {
    ai_generated: {
      chip: "AI-generated",
      title: "This image is AI-generated",
      sub: "Synthetic artefacts or provenance dominate the evidence. Treat this as a strong lead, not proof.",
    },
    real: {
      chip: "Camera capture",
      title: "This image is a real photo",
      sub: "Signals are consistent with camera-captured imagery; this does not rule out subtle edits.",
    },
    uncertain: {
      chip: "Inconclusive",
      title: "This image is inconclusive",
      sub: "Signals conflict or fall inside the uncertainty band. Review the evidence before acting.",
    },
  }[verdict.label];

  verdictChip.textContent = copy.chip;
  verdictTitle.textContent = copy.title;
  verdictSub.textContent = copy.sub;
  gaugeNumber.textContent = String(Math.round(verdict.ai_probability * 100));

  gaugeValue.style.strokeDashoffset = String(GAUGE_CIRCUMFERENCE);
  requestAnimationFrame(() => {
    gaugeValue.style.strokeDashoffset = String(GAUGE_CIRCUMFERENCE * (1 - verdict.ai_probability));
  });

  $("statConfidence").textContent = pct(verdict.confidence);
  $("statReliability").textContent = pct(verdict.reliability);
  $("statEngine").textContent = verdict.engine;
  $("statLatency").textContent = `${result.elapsed_ms} ms`;
}

function renderSignals(signals) {
  signalList.replaceChildren();
  if (!signals.length) {
    signalList.appendChild(el("p", "caption", "No evidence signals were produced."));
    return;
  }
  for (const signal of signals) {
    const row = el("div", "signal");
    const head = el("div", "signal-head");
    head.append(el("span", "signal-label", signal.label));
    head.append(el("span", "signal-score", pct(signal.score)));
    const track = el("div", "signal-track");
    const fill = el("div", "signal-fill");
    if (signal.score >= 0.62) fill.classList.add("hot");
    else if (signal.score >= 0.42) fill.classList.add("warn");
    fill.style.transform = "scaleX(0)";
    track.appendChild(fill);
    row.append(head, track);
    row.append(
      el(
        "p",
        "signal-detail",
        `${signal.detail} Weight ${Math.round(signal.weight * 100)}%, reliability ${Math.round(
          signal.reliability * 100
        )}%.`
      )
    );
    signalList.appendChild(row);
    requestAnimationFrame(() => {
      fill.style.transform = `scaleX(${signal.score})`;
    });
  }
}

function renderViewer(result) {
  const base = $("baseImage");
  base.src = result.explain.preview;
  base.alt = `Analyzed image: ${result.filename}`;
  heatImage.src = result.edit.heatmap;
  boxImage.src = result.edit.boxes;
  applyViewerLayers();

  const { edit } = result;
  const regionText = edit.regions.length === 1 ? "1 suspicious region" : `${edit.regions.length} suspicious regions`;
  $("editCaption").textContent =
    `${pct(edit.edited_probability)} estimated edit probability, ${regionText}, ${edit.method} method. ` +
    edit.note;
}

function applyViewerLayers() {
  heatImage.classList.toggle("visible", toggleHeat.checked);
  heatImage.style.opacity = toggleHeat.checked ? String(heatOpacity.value / 100) : "0";
  boxImage.classList.toggle("visible", toggleBoxes.checked);
}

function renderAttribution(result) {
  const list = $("attributionList");
  list.replaceChildren();
  const caption = $("attributionCaption");
  const items = result.attribution;

  if (!items.length) {
    list.appendChild(el("p", "caption", "No attribution signal available."));
    caption.textContent = "";
    return;
  }

  for (const item of items) {
    const row = el("div", "attribution-row");
    const head = el("div", "attribution-head");
    const name = el("span", "attribution-name", item.generator);
    name.append(
      el("span", "muted", item.known ? (item.source === "metadata" ? "metadata" : "model") : "open-set")
    );
    head.append(name);
    head.append(el("span", "attribution-conf", pct(item.confidence)));
    const track = el("div", "attribution-track");
    const fill = el("div", `attribution-fill${item.known ? "" : " unknown"}`);
    fill.style.transform = "scaleX(0)";
    track.appendChild(fill);
    row.append(head, track);
    list.appendChild(row);
    requestAnimationFrame(() => {
      fill.style.transform = `scaleX(${item.confidence})`;
    });
  }

  const top = items[0];
  if (!top.known) {
    caption.textContent =
      "Closest match only: the generator falls outside the known set, so this is reported as unknown.";
  } else if (top.source === "metadata") {
    caption.textContent =
      "Attribution comes from embedded provenance metadata (C2PA / text chunks), which can be spoofed.";
  } else {
    caption.textContent =
      "Attribution predicted by the trained open-set head; low-confidence results should be treated as unknown.";
  }
}

function renderExplain(explain) {
  $("spectrumImage").src = explain.spectrum;
  const card = $("gradcamCard");
  if (explain.gradcam) {
    $("gradcamImage").src = explain.gradcam;
    card.classList.remove("hidden");
  } else {
    $("gradcamImage").removeAttribute("src");
    card.classList.add("hidden");
  }
}

function renderMetadata(metadata) {
  const grid = $("metadataGrid");
  grid.replaceChildren();
  $("metaFormat").textContent = `${metadata.format}, ${metadata.width} × ${metadata.height}`;

  const entries = [
    ["Dimensions", `${metadata.width} × ${metadata.height}`],
    ["File size", humanSize(metadata.file_size_bytes)],
    ["EXIF", metadata.has_exif ? "present" : "none", metadata.has_exif ? "ok" : "warn"],
    ["Camera", metadata.camera || "not reported"],
    ["Software", metadata.software || "not reported"],
    ["Captured", metadata.captured_at || "not reported"],
    ["JPEG quality", metadata.jpeg_quality_estimate ? `~${metadata.jpeg_quality_estimate}` : "n/a"],
    ["C2PA / Content Credentials", metadata.has_c2pa ? "present" : "not found", metadata.has_c2pa ? "ok" : null],
    ["C2PA agent", metadata.c2pa_agent || "not reported"],
    ["Digital source type", metadata.c2pa_source_type || "not reported", metadata.synthetic_provenance ? "warn" : null],
    ["Synthetic provenance", metadata.synthetic_provenance ? "claimed" : "not claimed", metadata.synthetic_provenance ? "warn" : null],
    ["XMP", metadata.has_xmp ? "present" : "not found"],
    ["PNG text keys", metadata.png_text_keys.length ? metadata.png_text_keys.join(", ") : "none"],
    ["Generator hint", metadata.generator_hint || "not reported", metadata.generator_hint ? "warn" : null],
    ["Watermark marker", metadata.watermark_marker || "not found"],
  ];

  for (const [label, value, tone] of entries) {
    const item = el("div", "meta-item");
    item.appendChild(el("dt", null, label));
    const dd = el("dd", tone || null, value);
    item.appendChild(dd);
    grid.appendChild(item);
  }
}

function renderNotes(result) {
  const list = $("notesList");
  list.replaceChildren();
  const notes = [...new Set([...(result.notes || []), ...(result.metadata.notes || [])])].filter(Boolean);
  if (!notes.length) notes.push("No additional caveats were recorded for this analysis.");
  for (const note of notes) list.appendChild(el("li", null, note));
}

function renderReport(result) {
  $("shaValue").textContent = result.sha256;
  const distortionLabel = distortionSelect.options[distortionSelect.selectedIndex].textContent;
  const stamp = (() => {
    try {
      return new Date(result.created_at).toLocaleString();
    } catch {
      return result.created_at;
    }
  })();
  $("timing").textContent = `${distortionLabel}, analyzed in ${result.elapsed_ms} ms, ${stamp}`;
  $("stressBtn").disabled = false;
  $("downloadJson").disabled = false;
}

function resetStress() {
  $("stressResults").replaceChildren();
}

async function runStress() {
  if (!currentAnalysis || busy) return;
  const btn = $("stressBtn");
  const container = $("stressResults");
  const source = currentAnalysis.source;
  busy = true;
  analyzeBtn.disabled = true;
  btn.disabled = true;
  container.replaceChildren();
  const chips = [];

  for (const [, label] of STRESS_LEVELS) {
    const chip = el("div", "stress-chip pending");
    chip.appendChild(el("span", null, label));
    chip.appendChild(el("strong", null, "…"));
    container.appendChild(chip);
    chips.push(chip);
  }

  try {
    for (let index = 0; index < STRESS_LEVELS.length; index += 1) {
      const [key] = STRESS_LEVELS[index];
      const chip = chips[index];
      try {
        const result = await analyzeSource(source, key);
        const { verdict } = result;
        const tone = verdict.label === "ai_generated" ? "ai" : verdict.label;
        chip.className = `stress-chip is-${tone}`;
        chip.querySelector("strong").textContent = pct(verdict.ai_probability);
        chip.appendChild(
          el(
            "span",
            "stress-label",
            verdict.label === "ai_generated" ? "AI" : verdict.label === "real" ? "Real" : "Uncertain"
          )
        );
      } catch {
        chip.className = "stress-chip";
        chip.querySelector("strong").textContent = "error";
      }
    }
  } finally {
    busy = false;
    analyzeBtn.disabled = false;
    btn.disabled = false;
  }
}

function downloadJson() {
  if (!currentAnalysis) return;
  const { result } = currentAnalysis;
  const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  const base = result.filename.replace(/\.[^.]+$/, "").replace(/[^\w-]+/g, "_") || "image";
  anchor.href = url;
  anchor.download = `image-detector_${base}_${result.id.slice(0, 8)}.json`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

function renderIntakeResult(result) {
  const { verdict } = result;
  const kind =
    verdict.label === "ai_generated" ? "is-ai" : verdict.label === "real" ? "is-real" : "is-uncertain";
  const copy = { ai_generated: "AI-generated", real: "Real photo", uncertain: "Inconclusive" }[
    verdict.label
  ];
  intakeResult.classList.remove("is-ai", "is-real", "is-uncertain");
  intakeResult.classList.add(kind);
  intakeResultValue.textContent = copy;
  intakeResultMeta.textContent = `P(AI) ${pct(verdict.ai_probability)}, confidence ${pct(
    verdict.confidence
  )}`;
  intakeResult.classList.remove("hidden");
}

function hideIntakeResult() {
  intakeResult.classList.add("hidden");
  intakeResult.classList.remove("is-ai", "is-real", "is-uncertain");
}

function setPreview(file) {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  dzPreview.src = previewUrl;
  dzPreview.alt = `Preview of ${file.name}`;
  dzPreview.classList.remove("hidden");
  dropzone.classList.add("has-image");
}

function clearPreview() {
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }
  dzPreview.removeAttribute("src");
  dzPreview.alt = "Preview of the selected image";
  dzPreview.classList.add("hidden");
  dropzone.classList.remove("has-image");
}

function selectFile(file) {
  if (!file) return;
  const extensionOk = /\.(jpe?g|png|webp|bmp|tiff?)$/i.test(file.name);
  if (!ALLOWED_TYPES.includes(file.type) && !extensionOk) {
    showToast("Unsupported file type. Use JPEG, PNG, WEBP, BMP or TIFF.");
    return;
  }
  if (file.size > MAX_BYTES) {
    showToast("File exceeds the 15 MB limit.");
    return;
  }
  if (file.size === 0) {
    showToast("That file appears to be empty.");
    return;
  }
  selectedFile = file;
  fileName.textContent = file.name;
  fileSize.textContent = humanSize(file.size);
  fileRow.classList.remove("hidden");
  setPreview(file);
  hideIntakeResult();
  dropzone.querySelector(".dz-title").textContent = "Image ready";
  dropzone.querySelector(".dz-sub").textContent = "Click to choose a different image";
}

function clearFile() {
  selectedFile = null;
  fileInput.value = "";
  fileRow.classList.add("hidden");
  clearPreview();
  hideIntakeResult();
  dropzone.querySelector(".dz-title").textContent = "Drop an image here";
  dropzone.querySelector(".dz-sub").textContent = "or click to browse. JPEG, PNG, WEBP, TIFF. Up to 15 MB";
}

function resetAll() {
  currentAnalysis = null;
  clearFile();
  fileName.textContent = "";
  fileSize.textContent = "";
  $("stressBtn").disabled = true;
  $("downloadJson").disabled = true;
  results.classList.add("hidden");
  stopLoading();
  howGrid.classList.remove("hidden");
  resetStress();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

async function loadHealth() {
  const chip = $("engineChip");
  try {
    const health = await requestJson("/api/health");
    chip.classList.add("is-live");
    const parts = [health.detector === "heuristic" ? "heuristic engine" : health.detector];
    if (health.attribution) parts.push("attribution");
    $("engineChipText").textContent = parts.join(", ");
    chip.title =
      `Image Detector v${health.version}, localizer: ${health.localizer}, ` +
      `CLIP backbone ${health.clip_loaded ? "loaded" : "not loaded"} (${health.clip_backbone})`;
  } catch {
    chip.classList.add("is-warn");
    $("engineChipText").textContent = "engine offline";
  }
}

async function loadExamples() {
  try {
    const examples = await requestJson("/api/examples");
    exampleGrid.replaceChildren();
    for (const example of examples) {
      const card = document.createElement("button");
      card.type = "button";
      card.className = "example-card";
      card.title = `Analyze sample: ${example.label}`;
      const image = new Image();
      image.width = 1200;
      image.height = 900;
      image.src = example.url;
      image.alt = example.label;
      image.loading = "lazy";
      const tag = el("span", `example-tag kind-${example.kind}`, example.kind === "ai" ? "AI" : "Real");
      const name = el("span", "example-name", example.label);
      card.append(image, tag, name);
      card.addEventListener("click", () => {
        if (busy) return;
        runAnalysis({ kind: "example", name: example.name });
      });
      exampleGrid.appendChild(card);
    }
  } catch {
    exampleGrid.replaceChildren(el("p", "caption", "Samples could not be loaded. Upload your own image above."));
  }
}

fileInput.addEventListener("change", () => selectFile(fileInput.files?.[0] || null));
$("clearFile").addEventListener("click", clearFile);

$("uploadForm").addEventListener("submit", (event) => {
  event.preventDefault();
  if (busy) return;
  if (!selectedFile) {
    showToast("Choose an image or click a sample to analyze first.");
    return;
  }
  runAnalysis({ kind: "file", file: selectedFile });
});

dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    fileInput.click();
  }
});
dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("is-drag");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("is-drag"));
dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("is-drag");
  selectFile(event.dataTransfer?.files?.[0] || null);
});
window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("drop", (event) => event.preventDefault());

toggleHeat.addEventListener("change", applyViewerLayers);
toggleBoxes.addEventListener("change", applyViewerLayers);
heatOpacity.addEventListener("input", applyViewerLayers);

$("stressBtn").addEventListener("click", runStress);
$("downloadJson").addEventListener("click", downloadJson);
$("resetBtn").addEventListener("click", resetAll);

loadHealth();
loadExamples();
