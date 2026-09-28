/* TwinRobo docs playground: pre-rendered camera outputs, picked in the browser. */
(function () {
  const VIEWS = { compare: "Compare", side: "Side by side", diff: "Difference", depth: "Depth" };
  const STATUS = {
    estimated: "Estimated from public specs",
    measured: "Measured on a real unit",
    manufacturer_verified: "Verified by the manufacturer",
  };

  function el(tag, attrs, text) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
    if (text != null) e.textContent = text;
    return e;
  }

  function init(root) {
    if (root.dataset.ready) return;
    root.dataset.ready = "1";
    const base = new URL(root.dataset.base, location.href);
    fetch(new URL("manifest.json", base))
      .then((r) => r.json())
      .then((m) => run(root, base, m))
      .catch(() => (root.querySelector(".trp-note").textContent = "Could not load the playground."));
  }

  function run(root, base, manifest) {
    const $ = (s) => root.querySelector(s);
    // default: Isaac Sim, ray cast, side by side
    const isaac = manifest.scenes.findIndex((sc) => sc.simulator.startsWith("Isaac"));
    const state = { scene: Math.max(0, isaac), camera: null, method: "raycast", view: "side" };
    try {
      Object.assign(state, JSON.parse(sessionStorage.getItem("tr-playground") || "{}"));
    } catch (e) {}
    if (!manifest.scenes[state.scene]) state.scene = 0;
    const url = (cam, name) => new URL(`${cam.dir}/${name}.webp`, base).href;
    const scene = () => manifest.scenes[state.scene];
    const camera = () =>
      scene().cameras.find((c) => c.id === state.camera) || scene().cameras[0];

    function buttons(key, items) {
      const g = $(`.trp-group[data-key="${key}"]`);
      g.replaceChildren();
      for (const [value, label] of items) {
        const b = el("button", { type: "button", "data-value": value }, label);
        b.addEventListener("click", () => {
          state[key] = key === "scene" ? Number(value) : value;
          render();
        });
        g.append(b);
      }
    }
    buttons("scene", manifest.scenes.map((s, i) => [String(i), s.simulator]));
    buttons("method", Object.entries(manifest.methods));
    buttons("view", Object.entries(VIEWS));

    const frame = $(".trp-frame"), under = $(".trp-under"), over = $(".trp-over");
    const overImg = over.querySelector("img"), slider = $(".trp-slider"), diff = $(".trp-diff");
    const setSplit = () => {
      const v = slider.value;
      over.style.clipPath = `inset(0 0 0 ${v}%)`;
      frame.style.setProperty("--split", `${v}%`);
    };
    slider.addEventListener("input", setSplit);

    function load(src) {
      return new Promise((ok) => {
        const i = new Image();
        i.crossOrigin = "anonymous";
        i.onload = () => ok(i);
        i.src = src;
      });
    }

    async function drawDiff(cam) {
      const [a, b] = await Promise.all([load(url(cam, "pinhole")), load(url(cam, state.method))]);
      diff.width = a.naturalWidth;
      diff.height = a.naturalHeight;
      const ctx = diff.getContext("2d", { willReadFrequently: true });
      ctx.drawImage(a, 0, 0);
      const pa = ctx.getImageData(0, 0, diff.width, diff.height);
      ctx.drawImage(b, 0, 0);
      const pb = ctx.getImageData(0, 0, diff.width, diff.height);
      const d = pa.data, e = pb.data;
      for (let i = 0; i < d.length; i += 4) {
        const v = Math.min(255, 3 * ((Math.abs(d[i] - e[i]) + Math.abs(d[i + 1] - e[i + 1]) + Math.abs(d[i + 2] - e[i + 2])) / 3));
        e[i] = e[i + 1] = e[i + 2] = v;  // 3x amplified, white = different
      }
      ctx.putImageData(pb, 0, 0);
    }

    function info(cam) {
      const aside = $(".trp-info");
      aside.replaceChildren();
      aside.append(el("h3", {}, cam.label), el("code", {}, cam.id));
      const dl = el("dl");
      const fov = cam.fov_deg;
      const rows = [
        ["Sensor", cam.sensor],
        ["Lens", cam.lens],
        ["Field of view", `${fov.h}° H · ${fov.v}° V · ${fov.d}° D`],
        ["Focus", cam.focus_m ? `${cam.focus_m} m` : "infinity"],
        ["Catalog status", STATUS[cam.status] || cam.status],
      ];
      for (const [k, v] of rows) dl.append(el("dt", {}, k), el("dd", {}, v));
      aside.append(dl);
      const a = el("a", { href: `../catalog/` }, "See the catalog →");
      aside.append(a);
    }

    function grid() {
      const g = $(".trp-grid");
      g.replaceChildren();
      for (const cam of scene().cameras) {
        const b = el("button", { type: "button", class: "trp-card" });
        if (cam.id === camera().id) b.classList.add("is-active");
        const img = el("img", { src: url(cam, state.view === "depth" ? "depth" : state.method), alt: cam.label, loading: "lazy" });
        b.append(img, el("span", {}, cam.label));
        b.addEventListener("click", () => {
          state.camera = cam.id;
          render();
          frame.scrollIntoView({ behavior: "smooth", block: "nearest" });
        });
        g.append(b);
      }
    }

    function render() {
      const s = scene(), cam = camera();
      state.camera = cam.id;
      try { sessionStorage.setItem("tr-playground", JSON.stringify(state)); } catch (e) {}
      for (const g of root.querySelectorAll(".trp-group")) {
        const cur = String(state[g.dataset.key]);
        for (const b of g.children) b.classList.toggle("is-active", b.dataset.value === cur);
      }
      $('.trp-group[data-key="method"]').classList.toggle("is-muted", state.view === "depth");
      const w = state.view === "side" ? 2 * cam.size[0] : cam.size[0];
      frame.style.aspectRatio = `${w} / ${cam.size[1]}`;
      frame.dataset.view = state.view;
      const method = manifest.methods[state.method];
      const [l, r] = {
        compare: ["Pinhole", method],
        side: ["Pinhole", method],
        diff: [`|${method} − pinhole|`, ""],
        depth: ["Depth", ""],
      }[state.view];
      $(".trp-tag-l").textContent = l;
      $(".trp-tag-r").textContent = r;
      under.src = url(cam, state.view === "depth" ? "depth" : "pinhole");
      overImg.src = url(cam, state.method);
      diff.hidden = state.view !== "diff";
      if (state.view === "diff") drawDiff(cam);
      setSplit();
      const [near, far] = s.depth_m;
      $(".trp-note").textContent =
        `${s.label}: ${s.note}` + (state.view === "depth" ? ` Depth ${near}–${far} m, log scale.` : "");
      info(cam);
      grid();
    }
    render();
  }

  const boot = () => {
    const root = document.getElementById("tr-playground");
    if (root) init(root);
  };
  if (typeof document$ !== "undefined") document$.subscribe(boot);
  else document.addEventListener("DOMContentLoaded", boot);
})();
