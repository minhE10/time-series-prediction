/* app.jsx — shell: header, tabs, view routing */
const { useState: useStateApp } = React;

function App() {
  const TS = window.TS;
  const [tab, setTab] = useStateApp("forecast");

  const [st, setSt] = useStateApp({
    model: "itransformer", dataset: "sunspots", horizon: 12,
    target: TS.DSMAP["sunspots"].targets[0].key,
    windowIndex: -1, seqLen: 48, scaler: "standard",
  });

  const tabs = [
    { id: "forecast", label: "Forecast" },
    { id: "compare", label: "Compare" },
    { id: "datasets", label: "Datasets" },
    { id: "theory", label: "Theory" },
  ];

  return (
    <div className="shell">
      <header className="masthead">
        <div className="mast-left">
          <div className="kicker">Applied Statistics &amp; Experimental Design · Group 10</div>
          <h1 className="mast-title">Time Series Prediction</h1>
        </div>
        <div className="mast-right" />
      </header>

      <nav className="tabs">
        {tabs.map((t) => (
          <button key={t.id} type="button" className={"tab" + (tab === t.id ? " active" : "")} onClick={() => setTab(t.id)}>{t.label}</button>
        ))}
      </nav>

      <main className="main">
        {tab === "forecast" && <ForecastView st={st} setSt={setSt} />}
        {tab === "compare" && <CompareView st={st} />}
        {tab === "datasets" && <DatasetsView />}
        {tab === "theory" && <TheoryView />}
      </main>

      <footer className="footnote">
        © 2026 Group 10 - 168274 · Hanoi University of Science and Technology. All rights reserved.
      </footer>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
