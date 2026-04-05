"use client";

import { useState, useEffect, useCallback } from "react";
import { Card, CardHeader, Badge, StatBlock, Slider, Select, Button, Tabs, Modal } from "./components/ui";
import { ProbabilityBars, ConfidenceGauge, ClassMetricsBars, PredictionHistoryChart, FeatureImportanceChart } from "./components/charts";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api/v1";

const DEFAULTS = {
  Soil_Type               : "Clay",
  Soil_pH                 : 6.5,
  Soil_Moisture           : 35.0,
  Organic_Carbon          : 0.9,
  Electrical_Conductivity : 1.0,
  Temperature_C           : 27.0,
  Humidity                : 60.0,
  Rainfall_mm             : 500.0,
  Sunlight_Hours          : 7.5,
  Wind_Speed_kmh          : 10.0,
  Crop_Type               : "Wheat",
  Crop_Growth_Stage       : "Vegetative",
  Season                  : "Rabi",
  Irrigation_Type         : "Drip",
  Water_Source            : "Groundwater",
  Field_Area_hectare      : 5.0,
  Mulching_Used           : "No",
  Previous_Irrigation_mm  : 30.0,
  Region                  : "North",
};

const CATEGORICALS = {
  Soil_Type         : ["Clay", "Loamy", "Sandy", "Silt"],
  Crop_Type         : ["Cotton", "Maize", "Potato", "Rice", "Sugarcane", "Wheat"],
  Crop_Growth_Stage : ["Flowering", "Harvest", "Sowing", "Vegetative"],
  Season            : ["Kharif", "Rabi", "Zaid"],
  Irrigation_Type   : ["Canal", "Drip", "Rainfed", "Sprinkler"],
  Water_Source      : ["Groundwater", "Rainwater", "Reservoir", "River"],
  Mulching_Used     : ["No", "Yes"],
  Region            : ["Central", "East", "North", "South", "West"],
};

const NUMERICALS = [
  { name: "Soil_pH",                 label: "Soil pH",                  min: 4.8, max: 8.2,   step: 0.1  },
  { name: "Soil_Moisture",           label: "Soil Moisture (%)",        min: 8,   max: 65,    step: 0.5  },
  { name: "Organic_Carbon",          label: "Organic Carbon",           min: 0.3, max: 1.6,   step: 0.01 },
  { name: "Electrical_Conductivity", label: "Electrical Cond.",         min: 0.1, max: 3.5,   step: 0.1  },
  { name: "Temperature_C",           label: "Temperature (°C)",         min: 12,  max: 42,    step: 0.5  },
  { name: "Humidity",                label: "Humidity (%)",             min: 25,  max: 95,    step: 0.5  },
  { name: "Rainfall_mm",             label: "Rainfall (mm)",            min: 0,   max: 2500,  step: 10   },
  { name: "Sunlight_Hours",          label: "Sunlight Hours",           min: 4,   max: 11,    step: 0.5  },
  { name: "Wind_Speed_kmh",          label: "Wind Speed (km/h)",        min: 0.5, max: 20,    step: 0.5  },
  { name: "Field_Area_hectare",      label: "Field Area (ha)",          min: 0.3, max: 15,    step: 0.1  },
  { name: "Previous_Irrigation_mm",  label: "Prev. Irrigation (mm)",    min: 0,   max: 120,   step: 1    },
];

const TRAIN_MODELS = ["lgbm", "xgb", "catboost"];

export default function Dashboard() {
  const [form,        setForm]        = useState(DEFAULTS);
  const [tab,         setTab]         = useState("Crop & Soil");
  const [result,      setResult]      = useState(null);
  const [loading,     setLoading]     = useState(false);
  const [error,       setError]       = useState(null);
  const [metrics,     setMetrics]     = useState(null);
  const [modelInfo,   setModelInfo]   = useState(null);
  const [history,     setHistory]     = useState([]);
  const [trainModal,  setTrainModal]  = useState(false);
  const [apiKey,      setApiKey]      = useState("");
  const [trainModel,  setTrainModel]  = useState("lgbm");
  const [training,    setTraining]    = useState(false);

  const [trainLogs,   setTrainLogs]   = useState([]);
  const [trainResult,   setTrainResult]   = useState(null);
  const [importance,    setImportance]    = useState(null);
  const [trainHistory,  setTrainHistory]  = useState([]);
  const [predCount,     setPredCount]     = useState({ High: 0, Medium: 0, Low: 0 });
  const [mainTab,       setMainTab]       = useState("Overview");

  const fetchMeta = useCallback(async () => {
    try {
      const [mRes, iRes, fiRes, hRes] = await Promise.all([
        fetch(`${API}/metrics`),
        fetch(`${API}/metrics/model-info`),
        fetch(`${API}/metrics/feature-importance`),
        fetch(`${API}/train/history`),
      ]);
      if (mRes.ok)  setMetrics((await mRes.json()).data);
      if (fiRes.ok) setImportance((await fiRes.json()).data);
      if (hRes.ok)  setTrainHistory((await hRes.json()).data ?? []);
      if (iRes.ok) {
        const info = (await iRes.json()).data;
        setModelInfo(info);
        if (info?.model) setTrainModel(info.model);
      }
    } catch {}
  }, []);

  useEffect(() => { fetchMeta(); }, [fetchMeta]);

  function handleChange(e) {
    const { name, value } = e.target;
    setForm(f => ({ ...f, [name]: isNaN(value) || value === "" ? value : Number(value) }));
  }

  async function handlePredict() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API}/predict`, {
        method  : "POST",
        headers : { "Content-Type": "application/json" },
        body    : JSON.stringify(form),
      });
      if (res.status === 429) throw new Error("Rate limit reached. Wait a minute.");
      if (!res.ok) throw new Error((await res.json()).detail ?? "Prediction failed");
      const data = (await res.json()).data;
      setResult(data);
      setHistory(h => [...h.slice(-19), data]);
      setPredCount(c => ({ ...c, [data.irrigation_need]: (c[data.irrigation_need] ?? 0) + 1 }));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleTrain() {
    setTraining(true);
    setTrainResult(null);
    setTrainLogs([]);
    setTrainModal(false);

    try {
      const res = await fetch(`${API}/train`, {
        method  : "POST",
        headers : { "Content-Type": "application/json", "x-api-key": apiKey },
        body    : JSON.stringify({ model: trainModel }),
      });
      if (res.status === 401) throw new Error("Invalid API key.");
      if (!res.ok) throw new Error((await res.json()).message ?? "Failed to start training");

      const { data } = await res.json();
      setTrainLogs(["Job queued, starting..."]);

      const interval = setInterval(async () => {
        try {
          const sr    = await fetch(`${API}/train/${data.job_id}/status`);
          const sdata = (await sr.json()).data;

          setTrainLogs(sdata.logs ?? []);

          if (sdata.status === "done") {
            clearInterval(interval);
            setTraining(false);
            setTrainResult({ ok: true, score: sdata.score, run_id: sdata.run_id, model: sdata.model });
            fetchMeta();
          } else if (sdata.status === "failed") {
            clearInterval(interval);
            setTraining(false);
            setTrainResult({ ok: false, msg: sdata.error ?? "Training failed" });
          }
        } catch {
          clearInterval(interval);
          setTraining(false);
          setTrainResult({ ok: false, msg: "Lost connection to server" });
        }
      }, 2000);

    } catch (e) {
      setTrainResult({ ok: false, msg: e.message });
      setTraining(false);
    }
  }

  function dismissTrainToast() {
    setTrainResult(null);
    setTrainLogs([]);
  }

  const ba      = metrics?.balanced_accuracy;
  const report  = metrics?.classification_report;
  const totalPreds = predCount.High + predCount.Medium + predCount.Low;
  const mlflowRunUrl = modelInfo?.mlflow_uri && modelInfo?.run_id
    ? `${modelInfo.mlflow_uri}/#/experiments/1/runs/${modelInfo.run_id}`
    : null;

  return (
    <div className="h-screen flex flex-col overflow-hidden">

      {/* ── Topbar ── */}
      <header className="shrink-0 border-b border-[var(--border)] bg-[var(--surface)] px-5 h-12 flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <h1 className="text-sm font-bold tracking-tight">Irrigation MLOps</h1>
          {modelInfo?.model && (
            <span className="text-xs px-2 py-0.5 rounded bg-slate-100 font-mono text-slate-600">
              {modelInfo.model.toUpperCase()}
            </span>
          )}
          {modelInfo?.run_id && mlflowRunUrl ? (
            <a
              href={mlflowRunUrl}
              target="_blank"
              rel="noreferrer"
              className="hidden sm:block text-xs font-mono text-[var(--text-muted)] hover:text-[var(--accent)] transition-colors"
              title="View in MLflow"
            >
              #{modelInfo.run_id.slice(0, 7)} ↗
            </a>
          ) : modelInfo?.run_id ? (
            <span className="hidden sm:block text-xs font-mono text-[var(--text-muted)]">
              #{modelInfo.run_id.slice(0, 7)}
            </span>
          ) : null}
          <span className={`w-2 h-2 rounded-full ${modelInfo?.run_id ? "bg-green-500" : "bg-slate-300"}`} />
        </div>

        <div className="flex items-center gap-3">
          {totalPreds > 0 && (
            <div className="hidden sm:flex items-center gap-2 text-xs text-[var(--text-muted)]">
              <span>{totalPreds} predictions</span>
              <span className="text-red-500 font-mono">H:{predCount.High}</span>
              <span className="text-yellow-500 font-mono">M:{predCount.Medium}</span>
              <span className="text-green-500 font-mono">L:{predCount.Low}</span>
            </div>
          )}
          <Button variant="secondary" onClick={() => setTrainModal(true)} className="h-8 text-xs">
            ⚡ Run Training
          </Button>
        </div>
      </header>

      <div className="flex flex-1 min-h-0">

        {/* ── Sidebar ── */}
        <aside className="w-72 shrink-0 border-r border-[var(--border)] bg-[var(--surface)] flex flex-col">
          <div className="px-4 pt-3">
            <Tabs tabs={["Crop & Soil", "Environment"]} active={tab} onChange={setTab} />
          </div>

          <div className="flex-1 overflow-y-auto px-4 pb-4">
            {tab === "Crop & Soil" ? (
              <div className="flex flex-col gap-2.5">
                {Object.entries(CATEGORICALS).map(([name, opts]) => (
                  <Select key={name} name={name} label={name.replace(/_/g, " ")} options={opts} value={form[name]} onChange={handleChange} />
                ))}
              </div>
            ) : (
              <div className="flex flex-col gap-4">
                {NUMERICALS.map(({ name, label, min, max, step }) => (
                  <Slider key={name} name={name} label={label} min={min} max={max} step={step} value={form[name]} onChange={handleChange} />
                ))}
              </div>
            )}
          </div>

          <div className="shrink-0 p-3 border-t border-[var(--border)] flex flex-col gap-2">
            {error && (
              <p className="text-xs text-red-600 bg-red-50 border border-red-200 rounded-lg px-3 py-1.5">{error}</p>
            )}
            <Button onClick={handlePredict} loading={loading} className="w-full h-9">
              {loading ? "Predicting…" : "Predict Irrigation Need"}
            </Button>
          </div>
        </aside>

        {/* ── Main ── */}
        <main className="flex-1 min-h-0 flex flex-col bg-[var(--background)]">

          {/* Main tabs */}
          <div className="shrink-0 px-4 pt-3 border-b border-[var(--border)] bg-[var(--surface)]">
            <Tabs tabs={["Overview", "Model", "History"]} active={mainTab} onChange={setMainTab} />
          </div>

          <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-4">

            {/* ── Overview tab ── */}
            {mainTab === "Overview" && (
              <>
                {/* Row 1 — prediction result */}
                <div className="grid grid-cols-3 gap-4">
                  <Card className="flex flex-col items-center justify-center gap-2 min-h-[148px]">
                    {result ? (
                      <>
                        <span className="text-xs text-[var(--text-muted)] uppercase tracking-wider">Prediction</span>
                        <Badge label={result.irrigation_need} large />
                        <ConfidenceGauge value={result.confidence} />
                      </>
                    ) : (
                      <div className="flex flex-col items-center gap-2 text-center">
                        <span className="text-2xl">🌱</span>
                        <p className="text-xs text-[var(--text-muted)]">Configure inputs<br/>and click Predict</p>
                      </div>
                    )}
                  </Card>

                  <Card className="col-span-2">
                    <CardHeader title="Class Probabilities" subtitle="Model output distribution" />
                    {result ? (
                      <ProbabilityBars probabilities={result.probabilities} />
                    ) : (
                      <div className="flex items-center justify-center h-20 text-xs text-[var(--text-muted)]">
                        No prediction yet — fill the form and hit Predict
                      </div>
                    )}
                  </Card>
                </div>

                {/* Row 2 — session stats */}
                <div className="grid grid-cols-4 gap-4">
                  <Card>
                    <CardHeader title="Session Predictions" />
                    <StatBlock value={totalPreds || "—"} sub="this session" />
                  </Card>
                  {["High", "Medium", "Low"].map(cls => (
                    <Card key={cls}>
                      <CardHeader title={cls} subtitle="predictions" />
                      <StatBlock
                        value={predCount[cls] ?? "—"}
                        sub={totalPreds > 0 ? `${((predCount[cls] / totalPreds) * 100).toFixed(0)}% of total` : "no data"}
                      />
                    </Card>
                  ))}
                </div>

                {/* Row 3 — history chart */}
                <Card className="flex-1 min-h-0">
                  <CardHeader title="Prediction History" subtitle="Session probability trends" />
                  <PredictionHistoryChart history={history} />
                </Card>
              </>
            )}

            {/* ── Model tab ── */}
            {mainTab === "Model" && (
              <>
                {/* Row 1 — metric stats */}
                <div className="grid grid-cols-4 gap-4">
                  <Card>
                    <CardHeader title="Balanced Acc." />
                    {ba != null ? (
                      <StatBlock value={`${(ba * 100).toFixed(1)}%`} sub={modelInfo?.model ?? ""} />
                    ) : (
                      <div className="flex flex-col gap-1 pt-1">
                        <span className="text-lg font-bold text-[var(--text-muted)]">—</span>
                        <span className="text-xs text-[var(--text-muted)]">Train a model first</span>
                      </div>
                    )}
                  </Card>
                  {["High", "Medium", "Low"].map(cls => (
                    <Card key={cls}>
                      <CardHeader title={cls} subtitle="F1 · P · R" />
                      {report?.[cls] ? (
                        <StatBlock
                          value={`${(report[cls]["f1-score"] * 100).toFixed(1)}%`}
                          sub={`P ${(report[cls].precision * 100).toFixed(0)}  R ${(report[cls].recall * 100).toFixed(0)}`}
                        />
                      ) : (
                        <span className="text-xs text-[var(--text-muted)]">—</span>
                      )}
                    </Card>
                  ))}
                </div>

                {/* Row 2 — per-class bars + feature importance */}
                <div className="grid grid-cols-2 gap-4">
                  <Card>
                    <CardHeader title="Per-class Metrics" subtitle="Precision · Recall · F1" />
                    {report ? (
                      <ClassMetricsBars report={report} />
                    ) : (
                      <div className="flex items-center justify-center h-24 text-xs text-[var(--text-muted)]">
                        No metrics — run training to populate
                      </div>
                    )}
                  </Card>

                  <Card>
                    <CardHeader title="Feature Importance" subtitle="Normalised, top 12" />
                    {importance ? (
                      <FeatureImportanceChart importance={importance} />
                    ) : (
                      <div className="flex items-center justify-center h-24 text-xs text-[var(--text-muted)]">
                        No data — run training to populate
                      </div>
                    )}
                  </Card>
                </div>
              </>
            )}

            {/* ── History tab ── */}
            {mainTab === "History" && (
              <Card>
                <CardHeader title="Training Runs" subtitle="This session only — resets on restart" />
                {trainHistory.length === 0 ? (
                  <div className="flex flex-col items-center gap-2 py-10 text-xs text-[var(--text-muted)]">
                    <span className="text-2xl">⚡</span>
                    No training runs yet — click Run Training to start
                  </div>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="border-b border-[var(--border)] text-[var(--text-muted)]">
                          <th className="text-left py-2 pr-4 font-medium">Model</th>
                          <th className="text-left py-2 pr-4 font-medium">Status</th>
                          <th className="text-left py-2 pr-4 font-medium">Score</th>
                          <th className="text-left py-2 pr-4 font-medium">Run ID</th>
                          <th className="text-left py-2 pr-4 font-medium">Started</th>
                          <th className="text-left py-2 font-medium">Duration</th>
                        </tr>
                      </thead>
                      <tbody>
                        {trainHistory.map(job => {
                          const duration = job.started_at && job.finished_at
                            ? `${Math.round((new Date(job.finished_at) - new Date(job.started_at)) / 1000)}s`
                            : job.status === "running" ? "running…" : "—";
                          const started = job.started_at
                            ? new Date(job.started_at).toLocaleTimeString()
                            : "—";
                          const runUrl = modelInfo?.mlflow_uri && job.run_id
                            ? `${modelInfo.mlflow_uri}/#/experiments/1/runs/${job.run_id}`
                            : null;

                          return (
                            <tr key={job.job_id} className="border-b border-[var(--border)] last:border-0 hover:bg-slate-50">
                              <td className="py-2 pr-4 font-mono font-semibold">{job.model.toUpperCase()}</td>
                              <td className="py-2 pr-4">
                                <span className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium ${
                                  job.status === "done"    ? "bg-green-50 text-green-700" :
                                  job.status === "failed"  ? "bg-red-50 text-red-700" :
                                  job.status === "running" ? "bg-blue-50 text-blue-700" :
                                  "bg-slate-100 text-slate-600"
                                }`}>
                                  {job.status === "running" && <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />}
                                  {job.status}
                                </span>
                              </td>
                              <td className="py-2 pr-4 font-mono">
                                {job.score != null ? `${(job.score * 100).toFixed(2)}%` : job.error ? <span className="text-red-500 truncate max-w-[120px] block" title={job.error}>error</span> : "—"}
                              </td>
                              <td className="py-2 pr-4 font-mono text-[var(--text-muted)]">
                                {job.run_id ? (
                                  runUrl
                                    ? <a href={runUrl} target="_blank" rel="noreferrer" className="hover:text-[var(--accent)]">#{job.run_id.slice(0, 7)} ↗</a>
                                    : `#${job.run_id.slice(0, 7)}`
                                ) : "—"}
                              </td>
                              <td className="py-2 pr-4 text-[var(--text-muted)]">{started}</td>
                              <td className="py-2 text-[var(--text-muted)]">{duration}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                )}
              </Card>
            )}

          </div>
        </main>
      </div>

      {/* ── Train Modal ── */}
      <Modal
        open={trainModal}
        onClose={() => { setTrainModal(false); setTrainResult(null); setTrainLogs([]); }}
        title="Trigger Training"
      >
        <div className="flex flex-col gap-4">

          <div className="flex flex-col gap-1.5">
            <label className="text-xs text-[var(--text-muted)]">Model</label>
            <div className="flex gap-2">
              {TRAIN_MODELS.map(m => (
                <button
                  key={m}
                  onClick={() => setTrainModel(m)}
                  className={`flex-1 py-2 text-xs font-semibold rounded-lg border transition-colors ${
                    trainModel === m
                      ? "bg-[var(--accent)] text-[var(--accent-fg)] border-[var(--accent)]"
                      : "border-[var(--border)] text-[var(--text-muted)] hover:text-[var(--text-primary)]"
                  }`}
                >
                  {m.toUpperCase()}
                </button>
              ))}
            </div>
            {modelInfo?.model && (
              <p className="text-xs text-[var(--text-muted)]">
                Deployed: <span className="font-mono">{modelInfo.model}</span>
              </p>
            )}
          </div>

          <div className="flex flex-col gap-1">
            <label className="text-xs text-[var(--text-muted)]">API Key</label>
            <input
              type="password"
              placeholder="x-api-key…"
              value={apiKey}
              onChange={e => setApiKey(e.target.value)}
              onKeyDown={e => e.key === "Enter" && handleTrain()}
              className="w-full border border-[var(--border)] rounded-lg px-3 py-2 text-sm bg-[var(--background)] focus:outline-none focus:ring-1 focus:ring-[var(--accent)]/30"
            />
          </div>

          <div className="flex gap-2 justify-end pt-1">
            <Button variant="secondary" onClick={() => { setTrainModal(false); setTrainResult(null); setTrainLogs([]); }}>
              Cancel
            </Button>
            <Button onClick={handleTrain} className="min-w-[110px]">
              Start Training
            </Button>
          </div>
        </div>
      </Modal>

      {/* ── Training Toast ── */}
      {(training || trainResult) && (
        <div className="fixed bottom-4 right-4 z-50 w-80 rounded-xl border border-[var(--border)] bg-[var(--surface)] shadow-xl flex flex-col overflow-hidden">

          {/* Header */}
          <div className="flex items-center justify-between px-3 py-2 border-b border-[var(--border)]">
            <div className="flex items-center gap-2">
              {training
                ? <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                : <span className={`w-2 h-2 rounded-full ${trainResult?.ok ? "bg-green-500" : "bg-red-500"}`} />
              }
              <span className="text-xs font-semibold">
                {training ? `Training ${trainModel.toUpperCase()}…` : trainResult?.ok ? "Training complete" : "Training failed"}
              </span>
            </div>
            {!training && (
              <button onClick={dismissTrainToast} className="text-[var(--text-muted)] hover:text-[var(--text-primary)] text-xs leading-none">✕</button>
            )}
          </div>

          {/* Log feed */}
          <div className="bg-slate-950 px-3 py-2 flex flex-col gap-0.5 max-h-[120px] overflow-y-auto font-mono text-xs">
            {trainLogs.map((line, i) => (
              <span key={i} className="text-slate-300">
                <span className="text-slate-500 mr-1.5">›</span>{line}
              </span>
            ))}
            {training && (
              <span className="text-slate-500 mt-0.5">waiting...</span>
            )}
          </div>

          {/* Result footer */}
          {trainResult && (
            <div className={`px-3 py-2 text-xs font-medium ${trainResult.ok ? "text-green-700 bg-green-50" : "text-red-700 bg-red-50"}`}>
              {trainResult.ok
                ? `${trainResult.model?.toUpperCase()} · ${(trainResult.score * 100).toFixed(2)}% balanced acc · run #${trainResult.run_id?.slice(0, 7)}`
                : trainResult.msg
              }
            </div>
          )}
        </div>
      )}


    </div>
  );
}
