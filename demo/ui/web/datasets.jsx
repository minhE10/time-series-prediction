/* datasets.jsx — dataset preview: full-width panels stacked top-to-bottom */
const { useMemo: useMemoD } = React;

function DatasetPanel({ d }) {
  const TS = window.TS;
  const overviewVals = useMemoD(() => TS.overview(d.id, 260), [d.id]);
  const primary = d.targets[0];

  return (
    <div className="card ds-panel">
      <div className="card-head">
        <div>
          <div className="card-kicker">{d.type} · {d.freq} · {d.source}</div>
          <h3 className="card-title">{d.name} <span className={"ds-type inline " + (d.type === "Univariate" ? "uni" : "multi")}>{d.type === "Univariate" ? "uni" : "multi"}</span></h3>
        </div>
        <div className="ds-stat-strip">
          <div><span className="dss-num">{d.rows.toLocaleString()}</span><span className="dss-lbl">rows</span></div>
          <div><span className="dss-num">{d.span}</span><span className="dss-lbl">span</span></div>
          <div><span className="dss-num">{d.targets.length}</span><span className="dss-lbl">target{d.targets.length > 1 ? "s" : ""}</span></div>
        </div>
      </div>

      <p className="ds-blurb">{d.blurb}</p>

      {d.featureGroups && (
        <div className="ds-table">
          <table className="dt metric-table">
            <thead><tr><th>Feature group</th><th>Columns</th></tr></thead>
            <tbody>
              {d.featureGroups.map((g, i) => (
                <tr key={i}>
                  <td style={{ whiteSpace: "nowrap" }}>{g.label}</td>
                  <td className="muted-td" style={{ fontFamily: "var(--mono)", fontSize: "11px" }}>{g.cols.join(", ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="ds-overview-lbl">
        Overview — {primary.short} <span className="unit">/ {primary.unit}</span>
        <span className="muted" style={{ marginLeft: 8, fontSize: "10px", fontStyle: "italic" }}>(illustrative)</span>
      </div>
      <div className="ds-chart"><Sparkline values={overviewVals} color="var(--c-actual)" height={150} fill /></div>
    </div>
  );
}

function DatasetsView() {
  const TS = window.TS;
  return (
    <div className="view ds-stack">
      {TS.DATASETS.map((d) => <DatasetPanel key={d.id} d={d} />)}
    </div>
  );
}

Object.assign(window, { DatasetsView });
