/* forecast.jsx — single-model forecast view, full test set with dual-range slider */
const { useState: useStateF, useEffect: useEffectF, useMemo: useMemoF } = React;

function DualRangeSlider({ total, start, end, onStart, onEnd }) {
  if (!total) return null;
  const pct = (v) => (v / total) * 100;
  return (
    <div className="slider-wrap">
      <span className="slider-lbl">t{start + 1}</span>
      <div className="dual-slider">
        <div className="dual-slider-track" />
        <div className="dual-slider-fill" style={{ left: pct(start) + "%", right: (100 - pct(end)) + "%" }} />
        <input type="range" min={0} max={total} value={start}
          onChange={(e) => onStart(Math.min(parseInt(e.target.value), end - 1))} />
        <input type="range" min={0} max={total} value={end}
          onChange={(e) => onEnd(Math.max(parseInt(e.target.value), start + 1))} />
      </div>
      <span className="slider-lbl right">t{end} <span className="muted">/ {total}</span></span>
    </div>
  );
}

const RECON_MODES = [
  { key: "full",       label: "Non-overlap" },
  { key: "full_first", label: "First step"  },
  { key: "full_last",  label: "Last step"   },
];

function getFullEntry(model, dataset, horizon, target, recon) {
  const key = `${model}|${dataset}|${horizon}|${target}|${recon}`;
  return (window.PREDICTIONS && window.PREDICTIONS[key]) || null;
}

function ForecastView({ st, setSt }) {
  const TS = window.TS;
  const ds = TS.DSMAP[st.dataset];
  const [result, setResult] = useStateF(null);
  const [running, setRunning] = useStateF(false);
  const [rangeStart, setRangeStart] = useStateF(0);
  const [rangeEnd,   setRangeEnd]   = useStateF(0);
  const [showTable, setShowTable] = useStateF(false);
  const [recon, setRecon] = useStateF("full");

  const load = (reconMode) => {
    setRunning(true);
    const snap = { ...st };
    const rm = reconMode || recon;
    setTimeout(() => {
      const raw = getFullEntry(snap.model, snap.dataset, snap.horizon, snap.target, rm);
      const tgt = TS.DSMAP[snap.dataset].targets.find((t) => t.key === snap.target) || TS.DSMAP[snap.dataset].targets[0];
      let res;
      if (raw) {
        res = { ...raw, tgt, model: snap.model, dataset: snap.dataset, horizon: snap.horizon, target: snap.target, real: true };
      } else {
        res = TS.getForecast(snap.model, snap.dataset, snap.horizon, snap.target, -1);
      }
      setResult(res);
      setRangeStart(0);
      setRangeEnd(res.actual.length);
      setRunning(false);
    }, 200);
  };

  const changeRecon = (rm) => { setRecon(rm); load(rm); };

  useEffectF(() => { load(); /* eslint-disable-next-line */ }, []);
  useEffectF(() => { load(); /* eslint-disable-next-line */ }, [st.model, st.dataset, st.horizon, st.target]);

  const set = (patch) => setSt({ ...st, ...patch });

  const total = result ? result.actual.length : 0;
  const s = Math.max(0, Math.min(rangeStart, total));
  const e = Math.min(total, Math.max(rangeEnd, s + 1));

  const visActual    = useMemoF(() => result ? result.actual.slice(s, e) : [],    [result, s, e]);
  const visPredicted = useMemoF(() => result ? result.predicted.slice(s, e) : [], [result, s, e]);

  const unit = result ? result.tgt.unit : "";
  const series = [
    { name: "Actual",    color: "var(--c-actual)", points: visActual,    dots: false },
    { name: "Predicted", color: "var(--c-pred)",   points: visPredicted, dashed: true, dots: false },
  ];
  const m = result ? result.metrics : null;

  return (
    <div className="view">
      <div className="controlbar">
        <Select label="Model" value={st.model} onChange={(v) => set({ model: v })}
          options={TS.MODELS.map((mm) => ({ value: mm.id, label: mm.name, note: mm.family }))} />
        <Select label="Dataset" value={st.dataset} onChange={(v) => {
          const nd = TS.DSMAP[v]; set({ dataset: v, target: nd.targets[0].key });
        }} options={TS.DATASETS.map((d) => ({ value: d.id, label: d.name, note: d.type }))} />
        <Select label="Horizon" value={st.horizon} onChange={(v) => set({ horizon: v })}
          options={TS.HORIZONS.map((h) => ({ value: h, label: `${h} steps` }))} />
      </div>

      <div className="result-grid">
        <div className="card chart-card">
          <div className="card-head">
            <div>
              <div className="card-kicker">{TS.MMAP[st.model].name} · horizon {st.horizon} · full test set</div>
              <h3 className="card-title">{result ? result.tgt.short : ds.targets[0].short} <span className="unit">/ {unit}</span></h3>
            </div>
            {ds.targets.length > 1 && (
              <div className="target-pills">
                {ds.targets.map((t) => (
                  <button key={t.key} type="button" className={"pill" + (t.key === st.target ? " on" : "")} onClick={() => set({ target: t.key })}>{t.short}</button>
                ))}
              </div>
            )}
          </div>

          <Legend items={[{ name: "Actual", color: "var(--c-actual)" }, { name: "Predicted", color: "var(--c-pred)", dashed: true }]} />

          <div className="recon-toggle">
            {RECON_MODES.map((rm) => (
              <button key={rm.key} type="button" className={"mini-toggle" + (recon === rm.key ? " on" : "")} onClick={() => changeRecon(rm.key)}>{rm.label}</button>
            ))}
          </div>

          <DualRangeSlider total={total} start={s} end={e}
            onStart={setRangeStart} onEnd={setRangeEnd} />

          <div className={"chart-wrap" + (running ? " loading" : "")}>
            {visActual.length > 0 && <LineChart series={series} unit={unit} height={360} />}
          </div>
        </div>

        <div className="metrics-col">
          <div className="metrics-2x2">
            <MetricCard label="MAE"   value={m ? fmtNum(m.mae, unit) : "—"} sub="mean abs error" />
            <MetricCard label="RMSE"  value={m ? fmtNum(m.rmse, unit) : "—"} sub="root mean sq." />
            <MetricCard label="WMAPE" value={m ? (m.wmape * 100).toFixed(1) + "%" : "—"} sub="weighted MAPE" />
            <MetricCard label="R²"    value={m ? m.r2.toFixed(3) : "—"} sub="goodness of fit" accent={m && m.r2 > 0.8 ? "#2e6e6a" : null} />
          </div>
        </div>
      </div>

      {result && (
        <div className="card table-card">
          <button type="button" className="table-toggle" onClick={() => setShowTable(!showTable)}>
            <svg width="10" height="10" viewBox="0 0 10 10" style={{ transform: showTable ? "rotate(90deg)" : "none", transition: "transform .15s" }}><path d="M2 1l5 4-5 4" fill="none" stroke="currentColor" strokeWidth="1.4" /></svg>
            Forecast values ({e - s} of {total} steps)
          </button>
          {showTable && (
            <div className="table-scroll">
              <table className="dt">
                <thead><tr><th>step</th><th>actual</th><th>predicted</th><th>error</th></tr></thead>
                <tbody>
                  {visActual.map((a, i) => {
                    const p = visPredicted[i]; const err = a.v - p.v;
                    return <tr key={i}><td>{s + i + 1}</td><td>{fmtNum(a.v, unit)}</td><td>{fmtNum(p.v, unit)}</td><td className={err >= 0 ? "pos" : "neg"}>{err >= 0 ? "+" : ""}{fmtNum(err, unit)}</td></tr>;
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

Object.assign(window, { ForecastView, DualRangeSlider, RECON_MODES });
