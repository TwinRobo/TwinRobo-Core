/* TwinRobo docs, camera catalog: a browser over every catalog camera (assets/catalog.json)
   and a viewer of its pre-rendered views (assets/playground/manifest.json). */
(function () {
  const VIEWS = { compare: "Slider", side: "Side by side", diff: "Difference" };
  const METHODS = { psf: "PSF", raycast: "Ray cast" };  // short labels for the bar
  const OUTPUTS = {
    raw: ["Raw sensor", "The image as the sensor records it: the lens' distortion, blur and vignetting"],
    cal: ["Calibrated", "Undistorted with the camera's calibration, as a stereo SDK rectifies its streams; the lens blur stays"],
  };
  const KEY = "tr-catalog";

  function el(tag, attrs, text) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
    if (text != null) e.textContent = text;
    return e;
  }
  const fmt = (v, d = 1) => (v == null ? "–" : String(+(+v).toFixed(d)));

  function init(root) {
    if (root.dataset.ready) return;
    root.dataset.ready = "1";
    const base = new URL(root.dataset.base, location.href);
    const json = (u) => fetch(u).then((r) => (r.ok ? r.json() : Promise.reject(r.status)));
    Promise.all([
      json(new URL(root.dataset.catalog, location.href)),
      json(new URL("manifest.json", base)).catch(() => ({ scenes: [], methods: {} })),
    ])
      .then(([catalog, views]) => run(root, base, catalog, views))
      .catch(() => (root.querySelector(".trc-count").textContent = "Could not load the catalog."));
  }

  function run(root, base, catalog, views) {
    const $ = (s) => root.querySelector(s);
    const cams = catalog.cameras, byId = new Map(cams.map((c) => [c.id, c]));
    const name = (c) => [c.maker, c.product].filter(Boolean).join(" ") || c.id;

    // --- state: filters and viewer settings persist for the session; the camera is in the URL
    const isaac = views.scenes.findIndex((s) => s.simulator.startsWith("Isaac"));
    const state = { scene: Math.max(0, isaac), method: "raycast", output: "raw", view: "side", camera: null,
                    q: "", sort: "maker", maker: "", kind: "", status: "", previews: false, hfov: "" };
    try { Object.assign(state, JSON.parse(sessionStorage.getItem(KEY) || "{}")); } catch (e) {}
    if (!views.scenes[state.scene]) state.scene = 0;
    if (!(state.output in OUTPUTS) || !views.outputs) state.output = "raw";
    if (!(state.method in (views.methods || {}))) state.method = Object.keys(views.methods || {}).pop() || "psf";
    const fromHash = () => { const m = /cam=([^&]+)/.exec(location.hash); return m && decodeURIComponent(m[1]); };
    // default: the widest-angle camera with a preview (its lens shows most)
    const first = cams.filter((c) => c.views).sort((a, b) => (b.fov?.[0] ?? 0) - (a.fov?.[0] ?? 0))[0];
    state.camera = byId.has(fromHash()) ? fromHash() : byId.has(state.camera) ? state.camera : first?.id ?? null;
    const save = () => { try { sessionStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} };

    // --- filters
    const makers = [...new Set(cams.map((c) => c.maker).filter(Boolean))].sort();
    for (const m of makers) $(".trc-maker").append(el("option", { value: m }, m));
    for (const [k, v] of Object.entries(catalog.status || {})) $(".trc-status").append(el("option", { value: k }, v));
    const controls = { q: ".trc-q", sort: ".trc-sort", maker: ".trc-maker", kind: ".trc-kind",
                       status: ".trc-status", previews: ".trc-previews", hfov: ".trc-hfov" };
    for (const [k, sel] of Object.entries(controls)) {
      const c = $(sel);
      if (c.type === "checkbox") c.checked = !!state[k]; else c.value = state[k];
      c.addEventListener("input", () => { state[k] = c.type === "checkbox" ? c.checked : c.value; save(); list(); });
    }
    const words = (c) => [c.id, c.maker, c.product, c.sensor, c.shutter, c.mono ? "mono" : "color",
      c.focal_mm && `${c.focal_mm} mm`, c.status, ...c.modules].filter(Boolean).join(" ").toLowerCase();
    function filtered() {
      const q = state.q.trim().toLowerCase().split(/\s+/).filter(Boolean), min = +state.hfov || 0;
      const out = cams.filter((c) => q.every((w) => words(c).includes(w))
        && (!state.maker || c.maker === state.maker)
        && (!state.kind || (state.kind === "mono") === c.mono)
        && (!state.status || c.status === state.status)
        && (!state.previews || c.views)
        && (!min || (c.fov && c.fov[0] >= min)));
      const by = {
        maker: (a, b) => name(a).localeCompare(name(b)),
        hfov: (a, b) => (b.fov?.[0] ?? -1) - (a.fov?.[0] ?? -1),
        res: (a, b) => b.width * b.height - a.width * a.height,
        focal: (a, b) => (a.focal_mm ?? 1e9) - (b.focal_mm ?? 1e9),
      }[state.sort];
      return out.sort(by);
    }

    // --- views of a camera in the current scene
    const scene = () => views.scenes[state.scene];
    const viewOf = (id) => scene()?.cameras.find((v) => v.id === id);
    const url = (v, name) => new URL(`${v.dir}/${name}.webp`, base).href;
    // Calibrated only differs from raw for entries whose calibration has distortion.
    const output = () => (state.output === "cal" && byId.get(state.camera)?.distortion ? "cal" : "raw");
    const image = () => (output() === "cal" ? `${state.method}-cal` : state.method);  // the camera's image

    function list() {
      const box = $(".trc-list"), shown = filtered();
      $(".trc-count").textContent = shown.length === cams.length
        ? `${cams.length} cameras · ${makers.length} makers`
        : `${shown.length} of ${cams.length} cameras`;
      box.replaceChildren();
      let group = null;
      for (const c of shown) {
        if (state.sort === "maker" && c.maker !== group) {
          group = c.maker;
          box.append(el("div", { class: "trc-group" }, group || "Other"));
        }
        const b = el("button", { type: "button", class: "trc-item", role: "option", "data-id": c.id });
        const v = viewOf(c.id);
        const thumb = v ? el("img", { src: url(v, image()), alt: "", loading: "lazy" }) : el("span", { class: "trc-nothumb" });
        const text = el("span", { class: "trc-text" });
        text.append(el("b", {}, c.product || c.id),
          el("small", {}, [`${c.width}×${c.height}`, c.fov && `${fmt(c.fov[0], 0)}° H`, c.mono ? "mono" : null]
            .filter(Boolean).join(" · ")));
        b.append(thumb, text);
        if (c.nir) b.append(el("span", { class: "trc-badge is-ir", title: "Near-infrared camera" }, "IR"));
        if (c.status !== "estimated") b.append(el("span", { class: "trc-badge" }, "measured"));
        b.addEventListener("click", () => select(c.id));
        box.append(b);
      }
      if (!shown.length) box.append(el("p", { class: "trc-empty" }, "No camera matches."));
      mark();
    }
    function mark() {
      for (const b of root.querySelectorAll(".trc-item")) {
        const on = b.dataset.id === state.camera;
        b.classList.toggle("is-active", on);
        b.setAttribute("aria-selected", on);
      }
    }

    // --- viewer
    function buttons(key, items) {
      const g = $(`.trp-group[data-key="${key}"]`);
      g.replaceChildren();
      for (const [value, label, tip] of items) {
        const b = el("button", { type: "button", "data-value": value, ...(tip && { title: tip }) }, label);
        b.addEventListener("click", () => {
          state[key] = key === "scene" ? Number(value) : value;
          save(); show(); if (key !== "view") list();
        });
        g.append(b);
      }
    }
    buttons("scene", views.scenes.map((s, i) => [String(i), s.simulator]));
    buttons("method", Object.keys(views.methods || {}).map((m) => [m, METHODS[m] || views.methods[m]]));
    if (views.outputs) buttons("output", Object.entries(OUTPUTS).map(([k, [l, t]]) => [k, l, t]));
    else $('.trp-group[data-key="output"]').remove();
    buttons("view", Object.entries(VIEWS));

    const frame = $(".trp-frame"), under = $(".trp-under"), over = $(".trp-over");
    const overImg = over.querySelector("img"), slider = $(".trp-slider"), diff = $(".trp-diff");
    const setSplit = () => {
      over.style.clipPath = `inset(0 0 0 ${slider.value}%)`;
      frame.style.setProperty("--split", `${slider.value}%`);
    };
    slider.addEventListener("input", setSplit);
    // The stage has a fixed height: the image fits inside it, so the specs below never move.
    const stage = $(".trp-box");
    let aspect = 16 / 10;
    function fit() {
      const W = stage.clientWidth, H = stage.clientHeight;
      const w = Math.min(W, H * aspect);
      frame.style.width = `${w}px`;
      frame.style.height = `${w / aspect}px`;
    }
    new ResizeObserver(fit).observe(stage);
    // "Use this camera": a small panel with copyable code
    const use = $(".trp-usebtn"), pop = $(".trp-usepop");
    const openPop = (on) => { pop.hidden = !on; use.setAttribute("aria-expanded", on); };
    use.addEventListener("click", (e) => { e.stopPropagation(); openPop(pop.hidden); });
    pop.addEventListener("click", (e) => e.stopPropagation());
    document.addEventListener("click", () => openPop(false));
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") openPop(false); });
    for (const b of root.querySelectorAll(".trp-copy")) {
      b.addEventListener("click", async () => {
        const text = b.parentElement.querySelector("code").textContent;
        try { await navigator.clipboard.writeText(text); b.textContent = "Copied"; }
        catch (e) { b.textContent = "Select and copy"; }
        setTimeout(() => (b.textContent = "Copy"), 1500);
      });
    }
    const load = (src) => new Promise((ok) => { const i = new Image(); i.onload = () => ok(i); i.src = src; });
    async function drawDiff(v) {
      const [a, b] = await Promise.all([load(url(v, "pinhole")), load(url(v, image()))]);
      diff.width = a.naturalWidth; diff.height = a.naturalHeight;
      const ctx = diff.getContext("2d", { willReadFrequently: true });
      ctx.drawImage(a, 0, 0);
      const pa = ctx.getImageData(0, 0, diff.width, diff.height);
      ctx.drawImage(b, 0, 0);
      const pb = ctx.getImageData(0, 0, diff.width, diff.height), d = pa.data, e = pb.data;
      for (let i = 0; i < d.length; i += 4) {
        const x = Math.min(255, (Math.abs(d[i] - e[i]) + Math.abs(d[i + 1] - e[i + 1]) + Math.abs(d[i + 2] - e[i + 2])));
        e[i] = e[i + 1] = e[i + 2] = x;  // 3x amplified mean difference: white = different
      }
      ctx.putImageData(pb, 0, 0);
    }

    function info(c) {
      const aside = $(".trp-info");
      aside.replaceChildren();
      aside.append(el("h3", {}, name(c)));
      const dl = el("dl");
      const rows = [
        ["Sensor", [c.sensor, `${c.width}×${c.height}`, c.pitch_um && `${fmt(c.pitch_um, 2)} µm`, c.shutter && `${c.shutter} shutter`, c.mono && "monochrome"].filter(Boolean).join(" · ")],
        ["Lens", `${fmt(c.focal_mm, 2)} mm · f/${fmt(c.f_number)}${c.distortion ? " · distortion modeled" : ""}`],
        ["Field of view", c.fov ? `${fmt(c.fov[0])}° × ${fmt(c.fov[1])}° (pinhole)` : "from the lens file"],
        ["Focus", c.focus_m ? `${c.focus_m} m` : "infinity"],
        ["Status", (catalog.status || {})[c.status] || c.status],
      ];
      if (c.modules.length) rows.push(["Stereo module", c.modules.join(", ")]);
      for (const [k, v] of rows) {
        const fact = el("div", { class: "trp-fact" });
        fact.append(el("dt", {}, k), el("dd", {}, v));
        dl.append(fact);
      }
      aside.append(dl);
      if (c.nir) {
        aside.append(el("p", { class: "trp-irnote" },
          "Near-infrared camera. Its geometry and lens hold in the near IR, so these views show them; " +
          "they are rendered in visible light, as one channel. Near-IR appearance and the projector's " +
          "dots are not simulated yet."));
      }
      // the "Use this camera" panel
      $(".trp-usecode").textContent = `from twinrobo import CameraTwin\n\ncamera = CameraTwin.from_catalog("${c.id}")`;
      $(".trp-useyaml").href = c.yaml;
    }

    function show() {
      const c = byId.get(state.camera) || filtered()[0] || cams[0];
      state.camera = c.id;
      const v = viewOf(c.id), s = scene();
      const cal = root.querySelector('.trp-group[data-key="output"] button[data-value="cal"]');
      if (cal) {
        cal.disabled = !c.distortion;
        cal.title = c.distortion ? OUTPUTS.cal[1]
          : "This entry's calibration has no lens distortion, so its raw and calibrated images are the same";
      }
      for (const g of root.querySelectorAll(".trp-group")) {
        const cur = g.dataset.key === "output" ? output() : String(state[g.dataset.key]);
        for (const b of g.children) b.classList.toggle("is-active", b.dataset.value === cur);
      }
      $(".trp-none").hidden = !!v;
      root.querySelector(".trp-bar").classList.toggle("is-muted", !v);
      for (const e of [under, over, slider, diff, $(".trp-tag-l"), $(".trp-tag-r")]) e.style.visibility = v ? "" : "hidden";
      if (v) {
        const w = state.view === "side" ? 2 * v.size[0] : v.size[0];
        aspect = w / v.size[1];
        fit();
        frame.dataset.view = state.view;
        const method = views.methods[state.method] + (output() === "cal" ? " · calibrated" : "");
        const [l, r] = { compare: ["Pinhole", method], side: ["Pinhole", method], diff: [`|${method} − pinhole|`, ""] }[state.view];
        $(".trp-tag-l").textContent = l; $(".trp-tag-r").textContent = r;
        under.src = url(v, "pinhole");
        overImg.src = url(v, image());
        diff.hidden = state.view !== "diff";
        if (state.view === "diff") drawDiff(v);
        setSplit();
        $(".trp-note").textContent = `${s.label}: ${s.note}`;
      } else {
        aspect = 16 / 10;
        fit();
        $(".trp-note").textContent = "Its views have not been rendered for the docs yet; it works in every simulator all the same.";
      }
      info(c);
      mark();
      save();
    }

    function select(id, scroll) {
      if (!byId.has(id)) return;
      state.camera = id;
      history.replaceState(null, "", `#cam=${id}`);  // catalog ids are URL-safe (a-z 0-9 . - / _)
      show();
      if (scroll) root.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    window.addEventListener("hashchange", () => { const id = fromHash(); if (id && id !== state.camera) select(id); });
    for (const a of document.querySelectorAll("a[data-camera]")) {
      a.addEventListener("click", (e) => { e.preventDefault(); select(a.dataset.camera, true); });
    }
    // Arrow keys move through the list.
    $(".trc-list").addEventListener("keydown", (e) => {
      if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
      const items = [...root.querySelectorAll(".trc-item")], i = items.findIndex((b) => b.dataset.id === state.camera);
      const next = items[Math.max(0, Math.min(items.length - 1, i + (e.key === "ArrowDown" ? 1 : -1)))];
      if (next) { e.preventDefault(); select(next.dataset.id); next.focus(); next.scrollIntoView({ block: "nearest" }); }
    });

    list();
    show();
    const cur = root.querySelector(`.trc-item[data-id="${CSS.escape(state.camera)}"]`), box = $(".trc-list");
    if (cur) box.scrollTop = cur.offsetTop - box.offsetTop - box.clientHeight / 3;  // the list only, not the page
  }

  const boot = () => {
    const root = document.getElementById("tr-catalog");
    if (root) init(root);
  };
  if (typeof document$ !== "undefined") document$.subscribe(boot);
  else document.addEventListener("DOMContentLoaded", boot);
})();
