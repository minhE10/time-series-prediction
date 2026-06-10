/* ui.jsx — editorial UI primitives */
const { useState: useStateUI, useRef: useRefUI, useEffect: useEffectUI } = React;

/* Custom dropdown select */
function Select({ label, value, options, onChange, width }) {
  const [open, setOpen] = useStateUI(false);
  const boxRef = useRefUI(null);
  useEffectUI(() => {
    const h = (e) => { if (boxRef.current && !boxRef.current.contains(e.target)) setOpen(false); };
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, []);
  const cur = options.find((o) => o.value === value);
  return (
    <div className="field" style={width ? { width } : null}>
      {label && <label className="field-lbl">{label}</label>}
      <div className="select" ref={boxRef}>
        <button type="button" className={"select-btn" + (open ? " open" : "")} onClick={() => setOpen(!open)}>
          <span className="select-val">{cur ? cur.label : "—"}</span>
          <svg width="11" height="7" viewBox="0 0 11 7" className="caret"><path d="M1 1l4.5 4.5L10 1" fill="none" stroke="currentColor" strokeWidth="1.4" /></svg>
        </button>
        {open && (
          <div className="select-menu">
            {options.map((o) => (
              <button key={o.value} type="button"
                className={"opt" + (o.value === value ? " sel" : "") + (o.disabled ? " disabled" : "")}
                onClick={() => { if (o.disabled) return; onChange(o.value); setOpen(false); }}>
                <span className="opt-main">{o.label}</span>
                {o.note && <span className="opt-note">{o.note}</span>}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/* Segmented control */
function Segmented({ value, options, onChange }) {
  return (
    <div className="segmented">
      {options.map((o) => (
        <button key={o.value} type="button"
          className={"seg" + (o.value === value ? " active" : "")}
          onClick={() => onChange(o.value)}>{o.label}</button>
      ))}
    </div>
  );
}

/* Toggle chip (multi-select) */
function Chip({ active, color, onClick, children, disabled }) {
  return (
    <button type="button" disabled={disabled}
      className={"chip" + (active ? " on" : "") + (disabled ? " dis" : "")}
      onClick={onClick}>
      {color && <span className="chip-dot" style={{ background: active ? color : "transparent", borderColor: color }} />}
      {children}
    </button>
  );
}

/* Metric card */
function MetricCard({ label, value, sub, accent }) {
  return (
    <div className="metric">
      <div className="metric-lbl">{label}</div>
      <div className="metric-val" style={accent ? { color: accent } : null}>{value}</div>
      {sub && <div className="metric-sub">{sub}</div>}
    </div>
  );
}

/* Accordion (Advanced settings) */
function Accordion({ title, children, defaultOpen }) {
  const [open, setOpen] = useStateUI(!!defaultOpen);
  return (
    <div className={"accordion" + (open ? " open" : "")}>
      <button type="button" className="acc-head" onClick={() => setOpen(!open)}>
        <svg width="10" height="10" viewBox="0 0 10 10" className="acc-caret"><path d="M2 1l5 4-5 4" fill="none" stroke="currentColor" strokeWidth="1.4" /></svg>
        <span>{title}</span>
      </button>
      {open && <div className="acc-body">{children}</div>}
    </div>
  );
}

/* Legend item */
function Legend({ items }) {
  return (
    <div className="legend">
      {items.map((it, i) => (
        <span className="legend-item" key={i}>
          <span className="legend-mark" style={{ background: it.dashed ? "transparent" : it.color, borderBottom: it.dashed ? `2px dashed ${it.color}` : "none", height: it.dashed ? 0 : 3 }} />
          {it.name}
        </span>
      ))}
    </div>
  );
}

Object.assign(window, { Select, Segmented, Chip, MetricCard, Accordion, Legend });
