/* charts.jsx — editorial SVG charts: LineChart, BarChart, Sparkline */
const { useRef, useState, useEffect, useLayoutEffect, useCallback } = React;

/* measure a container's pixel width (responsive SVG with crisp interaction) */
function useWidth() {
  const ref = useRef(null);
  const [w, setW] = useState(720);
  useLayoutEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver((entries) => {
      const cw = entries[0].contentRect.width;
      if (cw > 0) setW(cw);
    });
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

function niceTicks(min, max, count) {
  const span = max - min || 1;
  const step0 = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const norm = step0 / mag;
  let step;
  if (norm < 1.5) step = 1; else if (norm < 3) step = 2; else if (norm < 7) step = 5; else step = 10;
  step *= mag;
  const start = Math.ceil(min / step) * step;
  const ticks = [];
  for (let v = start; v <= max + 1e-9; v += step) ticks.push(v);
  return ticks;
}

function fmtNum(v, unit) {
  const abs = Math.abs(v);
  let s;
  if (abs >= 1000) s = v.toLocaleString("en-US", { maximumFractionDigits: 0 });
  else if (abs >= 10) s = v.toFixed(0);
  else if (abs >= 1) s = v.toFixed(1);
  else s = v.toFixed(2);
  if (unit === "$") return "$" + s;
  return s;
}

/* ---- LineChart -------------------------------------------------------------
   series: [{ name, color, points:[{x,label,v}], dashed, muted, dots }]
   dividerX: numeric x where the forecast region begins (optional)            */
function LineChart({ series: rawSeries, height = 340, dividerX = null, unit = "", dividerLabel = "forecast →" }) {
  const [ref, W] = useWidth();
  const [hover, setHover] = useState(null); // {x, px}
  const pad = { l: 52, r: 18, t: 18, b: 30 };
  const H = height;
  const iw = Math.max(10, W - pad.l - pad.r);
  const ih = Math.max(10, H - pad.t - pad.b);

  // normalize points: accept either {x,...} or {i,...}
  const series = rawSeries.map((s) => ({
    ...s,
    points: s.points.map((p) => ({ x: p.x != null ? p.x : p.i, label: p.label, v: p.v })),
  }));

  const all = series.flatMap((s) => s.points);
  if (!all.length) return <div ref={ref} style={{ height }} />;
  const xs = all.map((p) => p.x), ys = all.map((p) => p.v);
  let xMin = Math.min(...xs), xMax = Math.max(...xs);
  let yMin = Math.min(...ys), yMax = Math.max(...ys);
  const yPad = (yMax - yMin) * 0.12 || 1;
  yMin -= yPad; yMax += yPad;
  if (xMax === xMin) xMax = xMin + 1;

  const sx = (x) => pad.l + ((x - xMin) / (xMax - xMin)) * iw;
  const sy = (v) => pad.t + (1 - (v - yMin) / (yMax - yMin)) * ih;

  const yticks = niceTicks(yMin, yMax, 4);
  // x ticks: sample ~6 labels from the union of points (sorted unique x)
  const uniqX = Array.from(new Set(all.map((p) => p.x))).sort((a, b) => a - b);
  const labelMap = {}; all.forEach((p) => { labelMap[p.x] = p.label; });
  const nXt = Math.min(6, uniqX.length);
  const xticks = Array.from({ length: nXt }, (_, k) => uniqX[Math.round((k / (nXt - 1 || 1)) * (uniqX.length - 1))]);

  const pathFor = (pts) => pts.map((p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(1)} ${sy(p.v).toFixed(1)}`).join(" ");

  const onMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    if (mx < pad.l - 4 || mx > pad.l + iw + 4) { setHover(null); return; }
    // nearest x in union
    let best = uniqX[0], bd = Infinity;
    for (const x of uniqX) { const d = Math.abs(sx(x) - mx); if (d < bd) { bd = d; best = x; } }
    setHover({ x: best });
  };

  const hoverRows = hover == null ? [] : series.map((s) => {
    const p = s.points.find((q) => q.x === hover.x);
    return p ? { name: s.name, color: s.color, v: p.v, label: p.label } : null;
  }).filter(Boolean);
  const hoverPx = hover != null ? sx(hover.x) : 0;
  const tipLeft = Math.min(Math.max(hoverPx + 12, pad.l), W - 168);

  return (
    <div ref={ref} style={{ position: "relative", width: "100%" }}>
      <svg width={W} height={H} onMouseMove={onMove} onMouseLeave={() => setHover(null)} style={{ display: "block", cursor: "crosshair" }}>
        {/* horizontal gridlines + y labels */}
        {yticks.map((t, i) => (
          <g key={"y" + i}>
            <line x1={pad.l} x2={pad.l + iw} y1={sy(t)} y2={sy(t)} stroke="var(--grid)" strokeWidth="1" />
            <text x={pad.l - 8} y={sy(t) + 3.5} textAnchor="end" className="ax-lbl">{fmtNum(t, unit)}</text>
          </g>
        ))}
        {/* x labels */}
        {xticks.map((x, i) => (
          <text key={"x" + i} x={sx(x)} y={H - 10} textAnchor="middle" className="ax-lbl">{labelMap[x]}</text>
        ))}
        {/* divider between context and forecast */}
        {dividerX != null && (
          <g>
            <line x1={sx(dividerX)} x2={sx(dividerX)} y1={pad.t} y2={pad.t + ih} stroke="var(--rule-md)" strokeWidth="1" strokeDasharray="3 3" />
            <text x={sx(dividerX) + 5} y={pad.t + 11} className="ax-lbl">{dividerLabel}</text>
          </g>
        )}
        {/* series */}
        {series.map((s, i) => (
          <path key={i} d={pathFor(s.points)} fill="none" stroke={s.color}
            strokeWidth={s.muted ? 1.4 : 2} strokeDasharray={s.dashed ? "5 4" : "none"}
            opacity={s.muted ? 0.55 : 1} strokeLinejoin="round" strokeLinecap="round" />
        ))}
        {/* dots on non-muted short series */}
        {series.filter((s) => s.dots).flatMap((s, si) =>
          s.points.map((p, i) => <circle key={si + "-" + i} cx={sx(p.x)} cy={sy(p.v)} r="2.6" fill={s.color} />)
        )}
        {/* crosshair */}
        {hover != null && (
          <g>
            <line x1={hoverPx} x2={hoverPx} y1={pad.t} y2={pad.t + ih} stroke="var(--ink)" strokeWidth="1" opacity="0.35" />
            {hoverRows.map((r, i) => <circle key={i} cx={hoverPx} cy={sy(r.v)} r="3.4" fill="var(--card)" stroke={r.color} strokeWidth="1.8" />)}
          </g>
        )}
      </svg>
      {hover != null && hoverRows.length > 0 && (
        <div className="tip" style={{ left: tipLeft, top: pad.t }}>
          <div className="tip-x">{hoverRows[0].label}</div>
          {hoverRows.map((r, i) => (
            <div className="tip-row" key={i}>
              <span className="tip-dot" style={{ background: r.color }} />
              <span className="tip-name">{r.name}</span>
              <span className="tip-val">{fmtNum(r.v, unit)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ---- BarChart (horizontal, leaderboard) ----------------------------------- */
function BarChart({ data, unit = "", lowerIsBetter = true }) {
  // data: [{label, value, color, sub}]
  const absMax = Math.max(...data.map((d) => Math.abs(d.value)));
  const max = absMax || 1;
  const ranked = [...data].sort((a, b) => lowerIsBetter ? a.value - b.value : b.value - a.value);
  const bestVal = ranked[0].value;
  return (
    <div className="barchart">
      {data.map((d, i) => {
        const best = d.value === bestVal;
        return (
          <div className="barrow" key={i}>
            <div className="barlabel">{d.label}{best && <span className="best-tag">best</span>}</div>
            <div className="bartrack">
              <div className="barfill" style={{ width: (d.value / max) * 100 + "%", background: d.color, opacity: best ? 1 : 0.72 }} />
            </div>
            <div className="barval" style={{ fontWeight: best ? 600 : 400 }}>{fmtNum(d.value, unit)}</div>
          </div>
        );
      })}
    </div>
  );
}

/* ---- Sparkline ------------------------------------------------------------- */
function Sparkline({ values, height = 44, color = "#16140f", fill = false }) {
  const [ref, W] = useWidth();
  if (!values || !values.length) return <div ref={ref} style={{ height }} />;
  const min = Math.min(...values), max = Math.max(...values);
  const sx = (i) => (i / (values.length - 1)) * (W - 2) + 1;
  const sy = (v) => height - 3 - ((v - min) / (max - min || 1)) * (height - 6);
  const d = values.map((v, i) => `${i ? "L" : "M"}${sx(i).toFixed(1)} ${sy(v).toFixed(1)}`).join(" ");
  const area = `${d} L${sx(values.length - 1)} ${height} L${sx(0)} ${height} Z`;
  return (
    <div ref={ref} style={{ width: "100%" }}>
      <svg width={W} height={height} style={{ display: "block" }}>
        {fill && <path d={area} fill={color} opacity="0.08" />}
        <path d={d} fill="none" stroke={color} strokeWidth="1.4" strokeLinejoin="round" />
      </svg>
    </div>
  );
}

Object.assign(window, { LineChart, BarChart, Sparkline, fmtNum, useWidth });
