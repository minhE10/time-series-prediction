/* theory.js — concise theory cards per model (Applied Statistics requirement) */
(function () {
  "use strict";
  window.TS = window.TS || {};
  window.TS.THEORY = [
    {
      id: "arima", name: "ARIMA", family: "Statistical", year: "1970",
      idea: "Autoregressive Integrated Moving Average. Differences the series to stationarity, then models it as a linear combination of past values (AR) and past forecast errors (MA).",
      formula: "\\phi(B)\\,(1-B)^{d}\\,y_t = \\theta(B)\\,\\varepsilon_t",
      params: ["p — AR order", "d — differencing", "q — MA order"],
      pros: ["Interpretable, well-understood", "Strong on stationary univariate signals", "No training infrastructure needed"],
      cons: ["Linear; misses complex patterns", "Manual order selection", "Weak on long, multivariate horizons"],
    },
    {
      id: "xgboost", name: "XGBoost", family: "Gradient Boosting", year: "2016",
      idea: "Forecasting recast as supervised regression over lagged features. An ensemble of regression trees is grown sequentially, each correcting the residuals of the last via gradient boosting.",
      formula: "\\hat{y} = \\sum_{k=1}^{K} f_k(\\mathbf{x}), \\quad f_k \\in \\mathcal{F}",
      params: ["n_estimators", "max_depth", "learning_rate"],
      pros: ["Handles non-linearities & interactions", "Robust with engineered lag features", "Fast, strong tabular baseline"],
      cons: ["No native temporal inductive bias", "Needs careful feature engineering", "Extrapolates poorly beyond training range"],
    },
    {
      id: "itransformer", name: "iTransformer", family: "Deep Learning", year: "2024",
      idea: "Inverts the Transformer: each variate (not each timestamp) becomes a token. Attention then models cross-variate correlations while a feed-forward net learns temporal representations.",
      formula: "\\mathbf{H} = \\mathrm{FFN}\\big(\\mathrm{Attn}(\\mathrm{Embed}(X^{\\top}))\\big)",
      params: ["d_model 32", "n_heads 4", "e_layers 2"],
      pros: ["State-of-the-art on multivariate sets", "Captures variate interactions", "Scales to long horizons"],
      cons: ["Needs GPU & training data", "Less interpretable", "Heavier than statistical baselines"],
    },
    {
      id: "timemixer", name: "TimeMixer", family: "Deep Learning", year: "2024",
      idea: "Decomposes the series into multiple scales, then mixes seasonal and trend components across resolutions with pure MLP blocks — no attention required.",
      formula: "\\hat{y} = \\mathrm{Mix}\\!\\Big( \\textstyle\\sum_{s} \\{\\mathbf{s}_s, \\mathbf{t}_s\\} \\Big)",
      params: ["d_model 32", "e_layers 2", "d_ff 64"],
      pros: ["Multi-scale seasonal/trend modelling", "Lightweight (MLP-only)", "Competitive accuracy"],
      cons: ["Sensitive to scale choice", "Still data-hungry", "Newer, fewer references"],
    },
    {
      id: "tsmamba", name: "TSMamba", family: "Deep Learning", year: "2024",
      idea: "Applies the Mamba selective state-space model to patches of the series. Linear-time sequence modelling with input-dependent gating, an efficient alternative to attention.",
      formula: "h_t = \\bar{A}\\,h_{t-1} + \\bar{B}\\,x_t, \\quad y_t = C\\,h_t",
      params: ["d_state 8", "patch_len 8", "expand 2"],
      pros: ["Linear-time in sequence length", "Strong long-range memory", "Efficient on long inputs"],
      cons: ["Implementation complexity", "Requires tuning of state size", "GPU recommended"],
    },
  ];
})();
