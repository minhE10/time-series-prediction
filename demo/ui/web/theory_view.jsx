/* theory_view.jsx — model theory cards */
function Formula({ tex }) {
  let html = tex;
  try {
    if (window.katex) html = window.katex.renderToString(tex, { displayMode: true, throwOnError: false });
  } catch (e) { /* fall back to raw */ }
  return <div className="theory-formula" dangerouslySetInnerHTML={{ __html: html }} />;
}

function TheoryView() {
  const TS = window.TS;
  return (
    <div className="view">
      <div className="theory-intro">
        <p>Five forecasting approaches span the project — from classical statistics to recent deep-learning architectures. Each is benchmarked on the same datasets and horizons.</p>
      </div>
      <div className="theory-grid">
        {TS.THEORY.map((m) => (
          <div key={m.id} className={"card theory-card" + (m.flagship ? " flagship" : "")}>
            <div className="theory-head">
              <span className="theory-mark" style={{ background: TS.MODEL_INK[m.id] }} />
              <div>
                <h3 className="theory-name">{m.name}</h3>
                <div className="theory-fam">{m.family} · {m.year}</div>
              </div>
              {m.flagship && <span className="flag-tag">flagship</span>}
            </div>
            <p className="theory-idea">{m.idea}</p>
            <Formula tex={m.formula} />
            <div className="theory-params">
              {m.params.map((p, i) => <span key={i} className="param-pill">{p}</span>)}
            </div>
            <div className="theory-procon">
              <div className="pc-col">
                <div className="pc-head pc-pro">Strengths</div>
                <ul>{m.pros.map((p, i) => <li key={i}>{p}</li>)}</ul>
              </div>
              <div className="pc-col">
                <div className="pc-head pc-con">Limitations</div>
                <ul>{m.cons.map((p, i) => <li key={i}>{p}</li>)}</ul>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

Object.assign(window, { TheoryView });
