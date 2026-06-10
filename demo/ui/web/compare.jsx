/* compare.jsx — model comparison view, full test set with dual-range slider */
const { useState: useStateC, useMemo: useMemoC, useEffect: useEffectC } = React;
const RECON_MODES = window.RECON_MODES || [
  { key: "full",       label: "Non-overlap" },
  { key: "full_first", label: "First step"  },
  { key: "full_last",  label: "Last step"   },
];

function metricCell(v, metric) {
  if (metric === "wmape") return (v * 100).toFixed(1) + "%";
  if (metric === "r2") return v.toFixed(3);
  return fmtNum(v);
}

function getFullRaw(model, dataset, horizon, target, recon) {
  const key = `${model}|${dataset}|${horizon}|${target}|${recon}`;
  return (window.PREDICTIONS && window.PREDICTIONS[key]) || null;
}

function CompareView({ st }) {
  const TS = window.TS;
  const [dataset, setDataset] = useStateC(st.dataset);
  const [horizon, setHorizon] = useStateC(st.horizon);
  const ds = TS.DSMAP[dataset];
  const [target, setTarget]     = useStateC(ds.targets[0].key);
  const [picked, setPicked]     = useStateC(["itransformer", "timemixer", "arima"]);
  const [leadMetric, setLeadMetric] = useStateC("rmse");
  const [rangeStart, setRangeStart] = useStateC(0);
  const [rangeEnd,   setRangeEnd]   = useStateC(0);
  const [recon, setRecon]           = useStateC("full");

  const changeDataset = (v) => { setDataset(v); setTarget(TS.DSMAP[v].targets[0].key); setRangeStart(0); setRangeEnd(0); };
  const changeHorizon = (v) => { setHorizon(v); setRangeStart(0); setRangeEnd(0); };
  const changeTarget  = (v) => { setTarget(v);  setRangeStart(0); setRangeEnd(0); };
  const togglePick = (id) => setPicked((p) => p.includes(id) ? (p.length > 1 ? p.filter((x) => x !== id) : p) : [...p, id]);

  const unit = (TS.DSMAP[dataset].targets.find((t) => t.key === target) || TS.DSMAP[dataset].targets[0]).unit;

  const modelRuns = useMemoC(() => picked.map((id) => {
    const raw = getFullRaw(id, dataset, horizon, target, recon);
    const fc  = raw
      ? { ...raw, tgt: TS.DSMAP[dataset].targets.find((t) => t.key === target) || TS.DSMAP[dataset].targets[0], real: true }
      : TS.getForecast(id, dataset, horizon, target, -1);
    return { id, fc };
  }), [picked, dataset, horizon, target, recon]);

  // Ground truth source: always the first model in canonical TS.MODELS order
  // that is currently picked — never depends on picked[] ordering.
  // Clip all series to the shortest actual length so x-axes align.
  const refRun = useMemoC(() =>
    TS.MODELS.map((mm) => modelRuns.find((r) => r.id === mm.id)).find(Boolean),
  [modelRuns]);

  const minLen = useMemoC(() =>
    modelRuns.length ? Math.min(...modelRuns.map((r) => r.fc.actual.length)) : 0,
  [modelRuns]);

  const total = minLen;

  // reset range whenever the common length changes (dataset/horizon/model set change)
  useEffectC(() => { setRangeStart(0); setRangeEnd(total); /* eslint-disable-next-line */ }, [total]);

  const effectiveEnd = rangeEnd === 0 ? total : rangeEnd;
  const s = Math.max(0, Math.min(rangeStart, total));
  const e = Math.min(total, Math.max(effectiveEnd, s + 1));

  const actualPts = useMemoC(() =>
    refRun ? refRun.fc.actual.slice(s, e) : [],
  [refRun, s, e]);

  const modelSeries = useMemoC(() => [
    { name: "Actual", color: "var(--c-actual)", points: actualPts, dots: false },
    ...modelRuns.map((r) => ({
      name: TS.MMAP[r.id].name,
      color: TS.MODEL_INK[r.id],
      points: r.fc.predicted.slice(s, Math.min(e, r.fc.predicted.length)),
      dashed: true,
    })),
  ], [modelRuns, actualPts, s, e]);

  const lowerBetter = leadMetric !== "r2";
  const leaderboard = modelRuns.map((r) => ({ label: TS.MMAP[r.id].name, value: r.fc.metrics[leadMetric], color: TS.MODEL_INK[r.id] }));

  const bestPerCol = useMemoC(() => {
    const cols = ["mae", "rmse", "wmape", "r2"]; const best = {};
    cols.forEach((c) => {
      const vals = modelRuns.map((r) => r.fc.metrics[c]);
      best[c] = c === "r2" ? Math.max(...vals) : Math.min(...vals);
    });
    return best;
  }, [modelRuns]);

  return (
    <div className="view">
      <div className="controlbar">
        <Select label="Dataset" value={dataset} onChange={changeDataset} options={TS.DATASETS.map((d) => ({ value: d.id, label: d.name, note: d.type }))} />
        <Select label="Horizon" value={horizon} onChange={changeHorizon} options={TS.HORIZONS.map((h) => ({ value: h, label: `${h} steps` }))} />
        {ds.targets.length > 1 && <Select label="Target" value={target} onChange={changeTarget} options={ds.targets.map((t) => ({ value: t.key, label: t.short }))} />}
      </div>
      <div className="model-chips">
        {TS.MODELS.map((mm) => (
          <Chip key={mm.id} active={picked.includes(mm.id)} color={TS.MODEL_INK[mm.id]} onClick={() => togglePick(mm.id)}>{mm.name}</Chip>
        ))}
      </div>

      <div className="result-grid">
        <div className="card chart-card">
          <div className="card-head">
            <div><div className="card-kicker">{TS.DSMAP[dataset].name} · horizon {horizon} · full test set</div>
            <h3 className="card-title">Predicted vs actual <span className="unit">/ {unit}</span></h3></div>
          </div>
          <Legend items={[{ name: "Actual", color: "var(--c-actual)" }, ...modelRuns.map((r) => ({ name: TS.MMAP[r.id].name, color: TS.MODEL_INK[r.id], dashed: true }))]} />

          <div className="recon-toggle">
            {RECON_MODES.map((rm) => (
              <button key={rm.key} type="button" className={"mini-toggle" + (recon === rm.key ? " on" : "")} onClick={() => setRecon(rm.key)}>{rm.label}</button>
            ))}
          </div>

          <DualRangeSlider total={total} start={s} end={e}
            onStart={setRangeStart} onEnd={setRangeEnd} />

          <div className="chart-wrap"><LineChart series={modelSeries} unit={unit} height={350} /></div>
        </div>

        <div className="card lead-card">
          <div className="card-head"><div><div className="card-kicker">Leaderboard</div><h3 className="card-title">Ranked by {leadMetric.toUpperCase()}</h3></div></div>
          <div className="lead-metric-toggle">
            {["mae", "rmse", "wmape", "r2"].map((mk) => (
              <button key={mk} type="button" className={"mini-toggle" + (leadMetric === mk ? " on" : "")} onClick={() => setLeadMetric(mk)}>{mk.toUpperCase()}</button>
            ))}
          </div>
          <BarChart data={leaderboard} unit={leadMetric === "wmape" || leadMetric === "r2" ? "" : unit} lowerIsBetter={lowerBetter} />
          <div className="lead-note">{lowerBetter ? "lower is better" : "higher is better"}</div>
        </div>
      </div>

      <div className="card table-card">
        <div className="card-head"><h3 className="card-title">Metrics — side by side <span className="muted" style={{ fontSize: "0.8em", fontWeight: 400 }}>(overall test set)</span></h3></div>
        <div className="table-scroll">
          <table className="dt metric-table">
            <thead><tr><th>Model</th><th>Family</th><th>MAE</th><th>RMSE</th><th>WMAPE</th><th>R²</th></tr></thead>
            <tbody>
              {modelRuns.map((r) => (
                <tr key={r.id}>
                  <td><span className="row-dot" style={{ background: TS.MODEL_INK[r.id] }} />{TS.MMAP[r.id].name}</td>
                  <td className="muted-td">{TS.MMAP[r.id].family}</td>
                  {["mae", "rmse", "wmape", "r2"].map((c) => (
                    <td key={c} className={r.fc.metrics[c] === bestPerCol[c] ? "best-cell" : ""}>{metricCell(r.fc.metrics[c], c)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

Object.assign(window, { CompareView });
