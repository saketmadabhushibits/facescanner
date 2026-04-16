/* FaceScanner front-end controller.
 * Handles camera capture, file upload, POSTing to /api/analyze, and
 * rendering the beauty + country result cards.
 */

(function () {
  "use strict";

  /* --- DOM refs -------------------------------------------------------- */
  const els = {
    tabs:        document.querySelectorAll(".tab"),
    video:       document.getElementById("video"),
    overlay:     document.getElementById("overlay"),
    preview:     document.getElementById("preview"),
    placeholder: document.getElementById("placeholder"),
    scrim:       document.getElementById("scrim"),
    status:      document.getElementById("status-msg"),
    btnStart:    document.getElementById("btn-start"),
    btnCapture:  document.getElementById("btn-capture"),
    btnRetake:   document.getElementById("btn-retake"),
    uploadWrap:  document.querySelector(".upload-wrap"),
    fileInput:   document.getElementById("file-input"),
    results:     document.getElementById("results"),
    ringScore:   document.getElementById("ring-score"),
    ringFg:      document.getElementById("ring-fg"),
    beautyBand:  document.getElementById("beauty-band"),
    metricsList: document.getElementById("metrics-list"),
    beautyNotes: document.getElementById("beauty-notes"),
    macroStrip:  document.getElementById("macro-strip"),
    countryList: document.getElementById("country-list"),
    countryPrim: document.getElementById("country-primary"),
  };

  const METRIC_META = [
    ["symmetry",       "Symmetry",        "scale"],
    ["golden_ratio",   "Golden ratio",    "sigma"],
    ["facial_thirds",  "Facial thirds",   "layout-grid"],
    ["eye_spacing",    "Eye spacing",     "eye"],
    ["jaw_balance",    "Jaw balance",     "shield"],
    ["lip_proportion", "Lip proportion",  "smile"],
  ];

  const MACRO_CLASS = {
    "White":       "macro-White",
    "Black":       "macro-Black",
    "East Asian":  "macro-East-Asian",
    "Indian":      "macro-Indian",
    "Other":       "macro-Other",
  };

  const COUNTRY_COLORS = {
    "White":      "linear-gradient(135deg, #fcd34d, #f59e0b)",
    "Black":      "linear-gradient(135deg, #fb7185, #e11d48)",
    "East Asian": "linear-gradient(135deg, #60a5fa, #2563eb)",
    "Indian":     "linear-gradient(135deg, #a78bfa, #7c3aed)",
    "Other":      "linear-gradient(135deg, #5eead4, #10b981)",
  };

  /* --- State ----------------------------------------------------------- */
  let stream = null;
  let currentTab = "camera";

  /* --- Helpers --------------------------------------------------------- */
  function renderIcons() { if (window.lucide) lucide.createIcons(); }

  function setStatus(msg, type) {
    if (!msg) { els.status.hidden = true; return; }
    els.status.hidden = false;
    els.status.textContent = msg;
    els.status.className = "status" + (type ? " " + type : "");
  }

  function scrim(on) { els.scrim.classList.toggle("on", !!on); }

  function stopStream() {
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  }

  function showStageVideo() {
    els.video.style.display = "block";
    els.preview.style.display = "none";
    els.placeholder.style.display = "none";
  }
  function showStagePreview(dataUrl) {
    els.video.style.display = "none";
    els.preview.src = dataUrl;
    els.preview.style.display = "block";
    els.placeholder.style.display = "none";
  }
  function showStagePlaceholder() {
    els.video.style.display = "none";
    els.preview.style.display = "none";
    els.placeholder.style.display = "flex";
  }

  function initials(name) {
    return name.split(/[\s\-\/]+/).slice(0, 2).map(w => w[0]).join("").toUpperCase();
  }

  /* --- Camera ---------------------------------------------------------- */
  async function startCamera() {
    setStatus(null);
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 960 }, facingMode: "user" },
        audio: false,
      });
      els.video.srcObject = stream;
      showStageVideo();
      els.btnStart.hidden = true;
      els.btnCapture.disabled = false;
      els.btnRetake.hidden = true;
    } catch (err) {
      setStatus("Camera access denied or unavailable. Try uploading a photo instead.", null);
    }
  }

  function captureFrame() {
    const canvas = document.createElement("canvas");
    const v = els.video;
    canvas.width = v.videoWidth; canvas.height = v.videoHeight;
    const ctx = canvas.getContext("2d");
    // Un-mirror for analysis (video is displayed mirrored).
    ctx.translate(canvas.width, 0); ctx.scale(-1, 1);
    ctx.drawImage(v, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.92);
  }

  /* --- Analyze --------------------------------------------------------- */
  async function analyze(imageDataUrl) {
    scrim(true); setStatus(null);
    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_base64: imageDataUrl }),
      });
      const data = await res.json();
      if (!data.ok) {
        setStatus(data.error || "Analysis failed.");
        scrim(false);
        return;
      }
      renderResults(data);
    } catch (err) {
      setStatus("Network error. Is the server running?");
    } finally {
      scrim(false);
    }
  }

  /* --- Render ---------------------------------------------------------- */
  function renderResults(data) {
    els.results.hidden = false;
    els.results.classList.add("fade-in");
    els.results.scrollIntoView({ behavior: "smooth", block: "start" });

    // Beauty
    const b = data.beauty;
    els.beautyBand.textContent = b.band;
    animateNumber(els.ringScore, 0, b.overall, 1100);
    const circumference = 2 * Math.PI * 68;
    els.ringFg.setAttribute("stroke-dasharray", circumference.toFixed(0));
    requestAnimationFrame(() => {
      els.ringFg.setAttribute(
        "stroke-dashoffset",
        (circumference * (1 - b.overall / 100)).toFixed(1)
      );
    });

    els.metricsList.innerHTML = "";
    METRIC_META.forEach(([key, label, icon]) => {
      const v = b[key];
      const li = document.createElement("li");
      li.className = "metric";
      li.innerHTML = `
        <span class="metric-label"><i data-lucide="${icon}"></i>${label}</span>
        <span class="metric-value">${v.toFixed(0)}</span>
        <span class="metric-bar"><span style="width:0%"></span></span>
      `;
      els.metricsList.appendChild(li);
      requestAnimationFrame(() => {
        li.querySelector(".metric-bar > span").style.width = v + "%";
      });
    });

    els.beautyNotes.innerHTML = "";
    b.notes.forEach(n => {
      const li = document.createElement("li");
      li.textContent = n;
      els.beautyNotes.appendChild(li);
    });

    // Country
    const macro = data.ethnicity.macro;
    els.macroStrip.innerHTML = "";
    macro.forEach(m => {
      const s = document.createElement("span");
      s.className = MACRO_CLASS[m.macro] || "macro-Other";
      s.style.width = "0%";
      s.title = `${m.label} — ${m.percentage}%`;
      els.macroStrip.appendChild(s);
      requestAnimationFrame(() => { s.style.width = m.percentage + "%"; });
    });

    els.countryList.innerHTML = "";
    const countries = data.ethnicity.countries;
    if (countries.length) {
      els.countryPrim.textContent = "Top: " + countries[0].country;
    }
    countries.forEach(c => {
      const li = document.createElement("li");
      li.className = "country-item";
      const bg = COUNTRY_COLORS[c.macro] || COUNTRY_COLORS["Other"];
      li.innerHTML = `
        <span class="country-icon" style="background:${bg}">${initials(c.country)}</span>
        <span class="country-text">
          <span class="country-name">${c.country}</span>
          <span class="country-region">${c.macro_label || ""}</span>
        </span>
        <span class="country-pct">${c.percentage.toFixed(1)}%</span>
      `;
      els.countryList.appendChild(li);
    });

    renderIcons();
  }

  function animateNumber(el, from, to, duration) {
    const start = performance.now();
    function step(t) {
      const p = Math.min(1, (t - start) / duration);
      const eased = 1 - Math.pow(1 - p, 3);
      el.textContent = (from + (to - from) * eased).toFixed(0);
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  /* --- Tab switching --------------------------------------------------- */
  function activateTab(name) {
    currentTab = name;
    els.tabs.forEach(t => t.classList.toggle("active", t.dataset.tab === name));
    const camMode = name === "camera";
    els.btnStart.hidden   = !camMode || !!stream;
    els.btnCapture.hidden = !camMode;
    els.btnRetake.hidden  = true;
    els.uploadWrap.hidden = camMode;
    if (!camMode) { stopStream(); showStagePlaceholder(); els.btnCapture.disabled = true; }
    setStatus(null);
  }

  /* --- Wire up --------------------------------------------------------- */
  els.tabs.forEach(t => t.addEventListener("click", () => activateTab(t.dataset.tab)));

  els.btnStart.addEventListener("click", startCamera);

  els.btnCapture.addEventListener("click", async () => {
    if (!stream) return;
    const dataUrl = captureFrame();
    showStagePreview(dataUrl);
    stopStream();
    els.btnStart.hidden = true;
    els.btnCapture.disabled = true;
    els.btnRetake.hidden = false;
    await analyze(dataUrl);
  });

  els.btnRetake.addEventListener("click", () => {
    showStagePlaceholder();
    els.results.hidden = true;
    els.btnRetake.hidden = true;
    if (currentTab === "camera") {
      els.btnStart.hidden = false;
      els.btnCapture.disabled = true;
    }
  });

  els.fileInput.addEventListener("change", (e) => {
    const f = e.target.files && e.target.files[0];
    if (!f) return;
    const reader = new FileReader();
    reader.onload = async () => {
      showStagePreview(reader.result);
      els.btnRetake.hidden = false;
      await analyze(reader.result);
    };
    reader.readAsDataURL(f);
  });

  renderIcons();
})();
