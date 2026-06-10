/* =============================================================================
   data.js  —  Demo data layer for the Time Series Prediction Demo
   -----------------------------------------------------------------------------
   This file fabricates realistic-looking forecast data so the UI can be shown
   without a Python backend. Everything funnels through getForecast(...) below.

   >>> TO PLUG IN REAL PREDICTIONS <<<
   Populate window.PREDICTIONS with entries keyed by
       `${model}|${dataset}|${horizon}|${target}|${windowIndex}`
   each holding:
       { history:   [{i, label, v}],     // input context (seq_len points)
         actual:    [{i, label, v}],      // ground-truth horizon
         predicted: [{i, label, v}],      // model forecast horizon
         metrics:   { mae, rmse, wmape, r2 } }
   getForecast() returns the real entry when present, otherwise synthesizes one.
   The synthetic generator is deterministic (seeded) so charts are stable.
   ========================================================================== */
(function () {
  "use strict";

  // ---- seeded RNG -----------------------------------------------------------
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function hashStr(s) {
    let h = 2166136261;
    for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
    return h >>> 0;
  }

  // ---- dataset catalog ------------------------------------------------------
  // freq drives the x-axis label format; targets carry display scale + unit.
  const DATASETS = [
    {
      id: "sunspots", name: "Sunspots", type: "Univariate", freq: "Monthly",
      rows: 3265, span: "1749 – 2024", source: "SILSO / WDC-SILSO, Brussels",
      blurb: "Sunspots are temporary phenomena on the Sun's photosphere — darker regions caused by concentrations of magnetic field flux that inhibit convection. Their count follows an approximately 11-year solar cycle. This is the canonical long-memory univariate benchmark in time-series research.",
      featureGroups: [
        { label: "Target", cols: ["Monthly Mean Total Sunspot Number"] },
        { label: "Cyclical", cols: ["month_sin", "month_cos"] },
      ],
      targets: [{ key: "Monthly Mean Total Sunspot Number", short: "Sunspots", unit: "count", base: 80, amp: 75, period: 132, noise: 0.10 }],
    },
    {
      id: "appliances_energy", name: "Appliances Energy", type: "Multivariate", freq: "10-min",
      rows: 19735, span: "Jan – May 2016", source: "UCI ML Repository",
      blurb: "Experimental data of appliances energy use in a low-energy building, logged every 10 minutes over ~4.5 months. House temperature and humidity were monitored with a ZigBee wireless sensor network across 9 rooms. Energy logged via m-bus meters; weather from Chievres Airport, Belgium (Reliable Prognosis). Two random variables were included to test regression robustness.",
      featureGroups: [
        { label: "Targets", cols: ["Appliances", "lights"] },
        { label: "Temperature", cols: ["T1–T9", "T_out"] },
        { label: "Humidity", cols: ["RH_1–RH_9", "RH_out"] },
        { label: "Weather", cols: ["Press_mm_hg", "Windspeed", "Visibility", "Tdewpoint"] },
        { label: "Cyclical", cols: ["hour_sin", "hour_cos"] },
      ],
      targets: [
        { key: "Appliances", short: "Appliances", unit: "Wh", base: 95, amp: 70, period: 144, noise: 0.42, floor: 10 },
        { key: "lights", short: "Lights", unit: "Wh", base: 16, amp: 14, period: 144, noise: 0.5, floor: 0 },
      ],
    },
    {
      id: "beijing_air_quality", name: "Beijing Air Quality", type: "Multivariate", freq: "Hourly",
      rows: 35064, span: "2013 – 2017", source: "UCI — Beijing Multi-Site",
      blurb: "Hourly PM2.5 and multi-pollutant readings from Beijing monitoring stations. PM2.5 refers to atmospheric particulate matter with diameter < 2.5 µm — a key indicator of air pollution used in environmental authority reports. Dataset includes 6 pollutants and 5 meteorological covariates with strong diurnal and seasonal cycles.",
      featureGroups: [
        { label: "Targets (pollutants)", cols: ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"] },
        { label: "Meteorology", cols: ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"] },
        { label: "Cyclical", cols: ["hour_sin", "hour_cos", "month_sin", "month_cos"] },
      ],
      targets: [
        { key: "PM2.5", short: "PM2.5", unit: "µg/m³", base: 78, amp: 55, period: 24, noise: 0.45, floor: 2 },
        { key: "PM10", short: "PM10", unit: "µg/m³", base: 104, amp: 65, period: 24, noise: 0.4, floor: 4 },
        { key: "SO2", short: "SO2", unit: "µg/m³", base: 18, amp: 12, period: 24, noise: 0.5, floor: 1 },
        { key: "NO2", short: "NO2", unit: "µg/m³", base: 50, amp: 26, period: 24, noise: 0.38, floor: 2 },
        { key: "CO", short: "CO", unit: "µg/m³", base: 1180, amp: 700, period: 24, noise: 0.42, floor: 100 },
        { key: "O3", short: "O3", unit: "µg/m³", base: 58, amp: 44, period: 24, noise: 0.4, floor: 1 },
      ],
    },
    {
      id: "hanoi_air_quality", name: "Hanoi Air Quality", type: "Multivariate", freq: "Hourly",
      rows: 26280, span: "2021 – 2024", source: "IQAir / OpenWeather",
      blurb: "Hourly AQI and pollutant levels from Hanoi monitoring stations. The dataset captures seasonal variations, hourly pollution patterns and the influence of weather covariates including temperature, humidity, wind speed, pressure, and UV index. Targets include PM2.5, PM10, AQI and four gas pollutants.",
      featureGroups: [
        { label: "Targets (pollutants)", cols: ["PM25", "PM10", "AQI", "CO", "NO2", "O3", "SO2"] },
        { label: "Weather", cols: ["Temperature", "Relative Humidity", "Pressure", "Wind Speed", "Precipitation", "Clouds", "UV Index"] },
        { label: "Cyclical", cols: ["hour_sin", "hour_cos", "month_sin", "month_cos"] },
      ],
      targets: [
        { key: "PM25", short: "PM2.5", unit: "µg/m³", base: 64, amp: 42, period: 24, noise: 0.46, floor: 3 },
        { key: "PM10", short: "PM10", unit: "µg/m³", base: 88, amp: 52, period: 24, noise: 0.42, floor: 5 },
        { key: "AQI", short: "AQI", unit: "index", base: 118, amp: 48, period: 24, noise: 0.3, floor: 10 },
        { key: "CO", short: "CO", unit: "µg/m³", base: 920, amp: 480, period: 24, noise: 0.4, floor: 120 },
        { key: "NO2", short: "NO2", unit: "µg/m³", base: 38, amp: 20, period: 24, noise: 0.4, floor: 2 },
        { key: "O3", short: "O3", unit: "µg/m³", base: 46, amp: 34, period: 24, noise: 0.42, floor: 1 },
        { key: "SO2", short: "SO2", unit: "µg/m³", base: 14, amp: 9, period: 24, noise: 0.5, floor: 1 },
      ],
    },
    {
      id: "bitcoin", name: "Bitcoin", type: "Univariate", freq: "Hourly",
      rows: 125833, span: "2014 – 2024", source: "Kaggle / CoinDesk",
      blurb: "Historical Bitcoin market data. Bitcoin is the longest-running and most well-known cryptocurrency, first released as open source in 2009 by the anonymous Satoshi Nakamoto. Transactions are verified and recorded in a public blockchain via SHA-256 cryptographic hashing. Daily open price captures the non-stationary, regime-shifting nature of crypto markets.",
      featureGroups: [
        { label: "Target", cols: ["Open (USD)"] },
      ],
      targets: [{ key: "Open", short: "Open (USD)", unit: "$", base: 32000, amp: 18000, period: 540, noise: 0.06, walk: true, floor: 3000 }],
    },
  ];

  // ---- model catalog --------------------------------------------------------
  // skill ∈ (0,1]; lower error multiplier = stronger model. available flips to
  // true once a real checkpoint / prediction file is wired in.
  const MODELS = [
    { id: "itransformer", name: "iTransformer", family: "Deep Learning", kind: "Inverted Transformer", err: 0.55, bias: 0.02, available: true },
    { id: "timemixer",    name: "TimeMixer",    family: "Deep Learning", kind: "Decomposition + MLP mixing", err: 0.70, bias: 0.03, available: true },
    { id: "tsmamba",      name: "TSMamba",      family: "Deep Learning", kind: "State-space (Mamba)", err: 0.74, bias: 0.04, available: true },
    { id: "xgboost",      name: "XGBoost",      family: "Gradient Boosting", kind: "Tree ensemble", err: 0.95, bias: 0.05, available: true },
    { id: "arima",        name: "ARIMA",        family: "Statistical", kind: "Autoregressive integrated MA", err: 1.25, bias: 0.08, available: true },
  ];

  const HORIZONS = [12, 24, 48];
  const SEQ_LEN = 48;

  const byId = (arr) => Object.fromEntries(arr.map((x) => [x.id, x]));
  const DSMAP = byId(DATASETS);
  const MMAP = byId(MODELS);

  // ---- base "ground truth" series for a dataset/target ----------------------
  function baseSeries(dataset, targetKey, n) {
    const ds = DSMAP[dataset];
    const tgt = ds.targets.find((t) => t.key === targetKey) || ds.targets[0];
    const rnd = mulberry32(hashStr(dataset + "::" + targetKey));
    const out = new Array(n);
    let walk = tgt.base;
    for (let i = 0; i < n; i++) {
      let v;
      if (tgt.walk) {
        // geometric-ish random walk with slow trend (bitcoin-like)
        const trend = 1 + 0.0009 * Math.sin((2 * Math.PI * i) / tgt.period);
        walk *= trend * (1 + (rnd() - 0.5) * tgt.noise * 0.9);
        walk = Math.max(tgt.floor || 0, walk);
        v = walk + tgt.amp * 0.4 * Math.sin((2 * Math.PI * i) / (tgt.period * 0.37));
      } else {
        const cyc = Math.sin((2 * Math.PI * i) / tgt.period);
        const cyc2 = 0.35 * Math.sin((2 * Math.PI * i) / (tgt.period * 0.27) + 1.1);
        const slow = 0.25 * Math.sin((2 * Math.PI * i) / (tgt.period * 6.3));
        const nz = (rnd() - 0.5) * 2 * tgt.amp * tgt.noise;
        v = tgt.base + tgt.amp * (cyc + cyc2 + slow) + nz;
      }
      if (tgt.floor != null) v = Math.max(tgt.floor, v);
      out[i] = v;
    }
    return { values: out, tgt };
  }

  // ---- x-axis labels --------------------------------------------------------
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  function labelFor(dataset, idx) {
    const freq = DSMAP[dataset].freq;
    if (freq === "Monthly") { const y = 1990 + Math.floor(idx / 12); return `${MONTHS[idx % 12]} ${y}`; }
    if (freq === "Daily") { const d = new Date(2021, 0, 1); d.setDate(d.getDate() + idx); return `${d.getDate()} ${MONTHS[d.getMonth()]}`; }
    if (freq === "Hourly") { const h = idx % 24; const day = Math.floor(idx / 24) % 30 + 1; return `D${day} ${String(h).padStart(2, "0")}:00`; }
    return `t${idx}`; // 10-min etc.
  }

  // ---- metrics --------------------------------------------------------------
  function computeMetrics(actual, predicted) {
    const n = actual.length;
    let se = 0, ae = 0, sa = 0, mean = 0;
    for (let i = 0; i < n; i++) mean += actual[i];
    mean /= n;
    let ssTot = 0;
    for (let i = 0; i < n; i++) {
      const e = actual[i] - predicted[i];
      se += e * e; ae += Math.abs(e); sa += Math.abs(actual[i]);
      ssTot += (actual[i] - mean) ** 2;
    }
    const mse = se / n;
    return {
      mae: ae / n,
      rmse: Math.sqrt(mse),
      wmape: ae / (sa + 1e-8),
      r2: ssTot > 0 ? 1 - se / ssTot : 0,
    };
  }

  // ---- core: build one forecast --------------------------------------------
  const TOTAL = 720; // synthetic series length
  function synth(model, dataset, horizon, target, windowIndex) {
    const m = MMAP[model];
    const { values, tgt } = baseSeries(dataset, target, TOTAL);
    // pick a window: windowIndex maps into the test region (last 20%)
    const testStart = Math.floor(TOTAL * 0.8);
    const maxStart = TOTAL - horizon - SEQ_LEN - 1;
    let start;
    if (windowIndex < 0) start = maxStart;
    else start = Math.min(testStart + windowIndex, maxStart);
    start = Math.max(SEQ_LEN, start);

    const history = [];
    for (let k = 0; k < SEQ_LEN; k++) {
      const i = start - SEQ_LEN + k;
      history.push({ i, label: labelFor(dataset, i), v: values[i] });
    }
    const actual = [], predicted = [];
    const rnd = mulberry32(hashStr([model, dataset, horizon, target, windowIndex].join("|")));
    const scale = tgt.amp * 0.5 + Math.abs(tgt.base) * 0.04;
    for (let k = 0; k < horizon; k++) {
      const i = start + k;
      const av = values[i];
      // error grows with horizon distance; model.err scales amplitude
      const growth = 0.45 + 0.55 * (k / Math.max(1, horizon - 1));
      const drift = m.bias * scale * (k / Math.max(1, horizon)) * (rnd() > 0.5 ? 1 : -1);
      const noise = (rnd() - 0.5) * 2 * scale * m.err * growth * 0.6;
      let pv = av + drift + noise;
      if (tgt.floor != null) pv = Math.max(tgt.floor, pv);
      actual.push({ i, label: labelFor(dataset, i), v: av });
      predicted.push({ i, label: labelFor(dataset, i), v: pv });
    }
    const metrics = computeMetrics(actual.map((p) => p.v), predicted.map((p) => p.v));
    return { history, actual, predicted, metrics, tgt, model, dataset, horizon, target, windowIndex, start };
  }

  function getForecast(model, dataset, horizon, target, windowIndex) {
    windowIndex = windowIndex == null ? -1 : windowIndex;
    const key = `${model}|${dataset}|${horizon}|${target}|${windowIndex}`;
    if (window.PREDICTIONS && window.PREDICTIONS[key]) {
      const real = window.PREDICTIONS[key];
      const metrics = real.metrics || computeMetrics(real.actual.map((p) => p.v), real.predicted.map((p) => p.v));
      return Object.assign({ tgt: (DSMAP[dataset].targets.find((t) => t.key === target) || DSMAP[dataset].targets[0]), model, dataset, horizon, target, windowIndex, real: true }, real, { metrics });
    }
    return synth(model, dataset, horizon, target, windowIndex);
  }

  // run-info string parts (mirrors the Gradio run_info)
  function runInfo(fc) {
    const m = MMAP[fc.model], ds = DSMAP[fc.dataset];
    return {
      artifact: `${fc.model}_${fc.dataset}_${fc.horizon}`,
      source: fc.real ? "predictions file" : "registered checkpoint",
      device: m.family === "Deep Learning" ? "cuda:0" : "cpu",
      rows: ds.rows, seqLen: SEQ_LEN,
      testWindows: Math.floor(ds.rows * 0.2) - SEQ_LEN - fc.horizon,
    };
  }

  // dataset overview series (long sparkline of primary target)
  function overview(dataset, points) {
    points = points || 220;
    const tgt = DSMAP[dataset].targets[0];
    const { values } = baseSeries(dataset, tgt.key, TOTAL);
    const step = TOTAL / points;
    const out = [];
    for (let k = 0; k < points; k++) out.push(values[Math.floor(k * step)]);
    return out;
  }

  function stats(dataset, target) {
    const { values } = baseSeries(dataset, target, TOTAL);
    let mn = Infinity, mx = -Infinity, sum = 0;
    for (const v of values) { mn = Math.min(mn, v); mx = Math.max(mx, v); sum += v; }
    const mean = sum / values.length;
    let varc = 0; for (const v of values) varc += (v - mean) ** 2;
    return { min: mn, max: mx, mean, std: Math.sqrt(varc / values.length) };
  }

  // model ink palette for compare overlays — CSS vars so themes can recolor
  const MODEL_INK = {
    itransformer: "var(--m-itransformer)",
    arima: "var(--m-arima)",
    xgboost: "var(--m-xgboost)",
    timemixer: "var(--m-timemixer)",
    tsmamba: "var(--m-tsmamba)",
  };

  window.TS = {
    DATASETS, MODELS, HORIZONS, SEQ_LEN, DSMAP, MMAP, MODEL_INK,
    getForecast, runInfo, overview, stats, computeMetrics, labelFor,
  };
})();
