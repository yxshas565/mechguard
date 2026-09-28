"use client";

import {
  Activity,
  ArrowRight,
  BrainCircuit,
  ChevronDown,
  CircleDot,
  CloudUpload,
  Eye,
  FileCode2,
  Layers3,
  Moon,
  Network,
  Play,
  RefreshCw,
  ShieldCheck,
  Sun,
  Upload,
  X,
  Zap,
} from "lucide-react";
import { DragEvent, useEffect, useRef, useState } from "react";

type Stage = "attest" | "watch" | "review";
type Layer = {
  id: Stage;
  eyebrow: string;
  title: string;
  description: string;
  icon: typeof ShieldCheck;
};

const layers: Layer[] = [
  {
    id: "attest",
    eyebrow: "TRAINING",
    title: "Attest",
    description:
      "Track how a model changes internally while it is being fine-tuned.",
    icon: ShieldCheck,
  },
  {
    id: "watch",
    eyebrow: "DEPLOYMENT",
    title: "Watch",
    description:
      "Monitor hidden-state representations across interacting agents.",
    icon: Eye,
  },
  {
    id: "review",
    eyebrow: "DECISION",
    title: "Review",
    description:
      "Turn internal signals into evidence that engineers can investigate.",
    icon: Activity,
  },
];

const evidence = [
  {
    label: "A001",
    title: "Training-time geometry",
    description:
      "45 checkpoints instrumented across a controlled LoRA fine-tuning run.",
    metric: "8.36×",
    metricLabel: "top singular value growth",
    target: "attest" as Stage,
  },
  {
    label: "B001",
    title: "Cross-agent representations",
    description:
      "Scenario-grouped evaluation of hidden-state representations across agents.",
    metric: "0.9677",
    metricLabel: "agent OOF AUROC",
    target: "watch" as Stage,
  },
  {
    label: "B003",
    title: "Layer stability",
    description:
      "Real activation measurements across layers 26–30 on Qwen3-32B.",
    metric: "1.00",
    metricLabel: "OOF AUROC",
    target: "watch" as Stage,
  },
];

export default function Home() {
  const [theme, setTheme] = useState<"light" | "dark">("dark");
  const [themeReady, setThemeReady] = useState(false);
  const [stage, setStage] = useState<Stage | null>(null);
  const [activeNode, setActiveNode] = useState<Stage | null>(null);
  const [useCase, setUseCase] = useState<number | null>(null);
  const [comparison, setComparison] = useState<number | null>(null);
  const [input, setInput] = useState("");
  const [fileName, setFileName] = useState("");
  const [result, setResult] = useState<any>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const workspaceRef = useRef<HTMLElement>(null);

  function applyTheme(next: "light" | "dark") {
    const root = document.documentElement;

    root.classList.toggle("dark", next === "dark");
    root.dataset.theme = next;

    if (next === "dark") {
      root.style.setProperty("--bg", "#0b0d0c");
      root.style.setProperty("--surface", "#111412");
      root.style.setProperty("--surface-2", "#171a18");
      root.style.setProperty("--text", "#f4f6f3");
      root.style.setProperty("--muted", "#929892");
      root.style.setProperty("--line", "rgba(255,255,255,.11)");
      root.style.setProperty("--line-strong", "rgba(255,255,255,.19)");
      root.style.setProperty("--accent", "#35d995");
      root.style.setProperty("--accent-soft", "rgba(53,217,149,.09)");
    } else {
      root.style.setProperty("--bg", "#f7f7f5");
      root.style.setProperty("--surface", "#ffffff");
      root.style.setProperty("--surface-2", "#f0f1ee");
      root.style.setProperty("--text", "#101210");
      root.style.setProperty("--muted", "#626762");
      root.style.setProperty("--line", "rgba(16,18,16,.13)");
      root.style.setProperty("--line-strong", "rgba(16,18,16,.22)");
      root.style.setProperty("--accent", "#08a968");
      root.style.setProperty("--accent-soft", "rgba(8,169,104,.09)");
    }

    root.style.colorScheme = next;
  }

  useEffect(() => {
    const saved = localStorage.getItem("mechguard-theme");

    const preferred: "light" | "dark" =
      saved === "light" || saved === "dark"
        ? saved
        : window.matchMedia("(prefers-color-scheme: light)").matches
          ? "light"
          : "dark";

    setTheme(preferred);
    applyTheme(preferred);
    setThemeReady(true);
  }, []);

  function toggleTheme() {
    const root = document.documentElement;
    const current = root.dataset.theme === "dark" ? "dark" : "light";
    const next = current === "dark" ? "light" : "dark";

    // Update React state
    setTheme(next);

    // Persist
    localStorage.setItem("mechguard-theme", next);

    // Update DOM state
    root.dataset.theme = next;
    root.classList.toggle("dark", next === "dark");
    root.style.colorScheme = next;

    // Force the entire document canvas
    const bg = next === "dark" ? "#0b0d0c" : "#f7f7f5";
    const fg = next === "dark" ? "#f4f6f3" : "#101210";

    root.style.setProperty("background-color", bg, "important");
    root.style.setProperty("color", fg, "important");

    document.body.style.setProperty("background-color", bg, "important");
    document.body.style.setProperty("color", fg, "important");

    // Force the actual MechGuard canvas
    const page = document.querySelector(".mechguard-page") as HTMLElement | null;

    if (page) {
      page.classList.toggle("theme-dark", next === "dark");
      page.classList.toggle("theme-light", next === "light");
      page.style.setProperty("background-color", bg, "important");
      page.style.setProperty("color", fg, "important");
    }
  }

  function scrollToWorkspace(nextStage?: Stage) {
    if (nextStage) setStage(nextStage);
    workspaceRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function chooseFile(file?: File) {
    if (!file) return;
    setFileName(file.name);
    setError("");
    const reader = new FileReader();
    reader.onload = () => setInput(String(reader.result ?? ""));
    reader.readAsText(file);
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    chooseFile(event.dataTransfer.files?.[0]);
  }

  function clearInput() {
    setInput("");
    setFileName("");
    setResult(null);
    setError("");
  }

  async function runAnalysis() {
    if (!input.trim()) {
      setError("Add a payload or upload a file first.");
      return;
    }

    setRunning(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          stage: stage ?? "attest",
          input,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data?.error || "Analysis failed.");
      }

      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed.");
    } finally {
      setRunning(false);
    }
  }

  return (
    <main className={`mechguard-page ${theme === "dark" ? "theme-dark" : "theme-light"}`}>
      <button
        className="mechguard-theme-toggle"
        onClick={toggleTheme}
        aria-label="Toggle theme"
        type="button"
        disabled={!themeReady}
      >
        {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
      </button>

      <section className="hero">
        <div className="hero-grid" />

        <div className="hero-copy">
          <div className="eyebrow">
            <CircleDot size={11} />
            MECHGUARD / INTERNAL AI OBSERVABILITY
          </div>

          <h1>
            Your AI can
            <br />
            <span>look fine.</span>
            <br />
            While changing underneath.
          </h1>

          <p className="hero-subtitle">
            MechGuard instruments the signals inside fine-tuned and
            multi-agent AI systems — before failures become visible at the
            boundary.
          </p>

          <div className="hero-actions">
            <button
              className="primary-button"
              onClick={() => scrollToWorkspace()}
              type="button"
            >
              Run the evidence
              <ArrowRight size={16} />
            </button>

            <a className="text-link" href="#architecture">
              See how it works
              <ChevronDown size={15} />
            </a>
          </div>
        </div>

        <div className="hero-visual">
          <div className="hero-visual-label">
            <span>INTERNAL SIGNAL PATH</span>
            <span>LIVE MODEL STATE</span>
          </div>

          <svg
            className="signal-svg"
            viewBox="0 0 620 340"
            role="img"
            aria-label="Animated internal signal paths"
          >
            <defs>
              <linearGradient id="heroSignal" x1="0%" x2="100%">
                <stop offset="0%" />
                <stop offset="50%" />
                <stop offset="100%" />
              </linearGradient>
            </defs>

            <path
              id="heroPathA"
              d="M40 170 C150 40 215 300 310 170 S470 40 580 170"
              fill="none"
              stroke="currentColor"
              strokeOpacity=".18"
              strokeWidth="1"
            />

            <path
              id="heroPathB"
              d="M40 230 C145 110 220 330 310 205 S470 100 580 220"
              fill="none"
              stroke="currentColor"
              strokeOpacity=".12"
              strokeWidth="1"
            />

            <circle cx="40" cy="170" r="5" fill="currentColor" />
            <circle cx="310" cy="170" r="5" fill="currentColor" />
            <circle cx="580" cy="170" r="5" fill="currentColor" />

            <circle r="3" fill="currentColor">
              <animateMotion dur="3.2s" repeatCount="indefinite">
                <mpath href="#heroPathA" />
              </animateMotion>
            </circle>

            <circle r="2.5" fill="currentColor">
              <animateMotion dur="4.5s" repeatCount="indefinite">
                <mpath href="#heroPathB" />
              </animateMotion>
            </circle>
          </svg>

          <div className="hero-node-labels">
            <span>MODEL</span>
            <span>INTERNAL STATE</span>
            <span>OBSERVATION</span>
          </div>
        </div>
      </section>

      <section className="product-section">
        <div className="section-heading">
          <div>
            <div className="eyebrow">THE PRODUCT</div>
            <h2>One layer. Three jobs.</h2>
          </div>
          <p>
            A single observability layer spanning training, deployment, and
            engineering review.
          </p>
        </div>

        <div className="product-grid">
          {layers.map((layer, index) => {
            const Icon = layer.icon;

            return (
              <button
                key={layer.id}
                className={`product-card ${stage === layer.id ? "is-selected" : ""}`}
                onClick={() => scrollToWorkspace(layer.id)}
                type="button"
              >
                <div className="product-card-top">
                  <span>0{index + 1}</span>
                  <Icon size={18} />
                </div>
                <div>
                  <div className="eyebrow">{layer.eyebrow}</div>
                  <h3>{layer.title}</h3>
                  <p>{layer.description}</p>
                </div>
                <ArrowRight className="card-arrow" size={16} />
              </button>
            );
          })}
        </div>
      </section>

      <section className="architecture-section" id="architecture">
        <div className="section-heading">
          <div>
            <div className="eyebrow">SYSTEM ARCHITECTURE</div>
            <h2>A signal flows through the system.</h2>
          </div>
          <p>
            Instrument the model, measure internal change, surface evidence,
            then review what needs attention.
          </p>
        </div>

        <div className="architecture-shell">
          <div className="architecture-map">
            {[
              ["instrument", "Instrument", Layers3],
              ["measure", "Measure", Activity],
              ["surface", "Surface", Network],
              ["review", "Review", FileCode2],
            ].map(([id, title, Icon], index) => {
              const selected =
                activeNode ===
                (id === "instrument"
                  ? "attest"
                  : id === "surface"
                    ? "watch"
                    : id === "review"
                      ? "review"
                      : null);

              const C = Icon as typeof Layers3;

              return (
                <div key={id as string} className="architecture-step">
                  <button
                    className={`system-node ${selected ? "is-selected" : ""}`}
                    type="button"
                    onClick={() => {
                      if (id === "instrument") setActiveNode("attest");
                      if (id === "measure") setActiveNode(null);
                      if (id === "surface") setActiveNode("watch");
                      if (id === "review") setActiveNode("review");
                    }}
                  >
                    <span className="node-index">0{index + 1}</span>
                    <C size={18} />
                    <strong>{title as string}</strong>
                  </button>

                  {index < 3 && <ArrowRight className="pipeline-arrow" size={16} />}
                </div>
              );
            })}
          </div>

          <div className="architecture-detail">
            {activeNode === null ? (
              <div className="architecture-empty">
                <Zap size={17} />
                <span>Hover or select a system stage to inspect it.</span>
              </div>
            ) : (
              <div>
                <div className="eyebrow">SELECTED SIGNAL</div>
                <h3>
                  {activeNode === "attest"
                    ? "Training-time internal change"
                    : activeNode === "watch"
                      ? "Cross-agent hidden representations"
                      : "Evidence for engineering review"}
                </h3>
                <p>
                  {activeNode === "attest"
                    ? "Measure how weight-space geometry evolves across fine-tuning checkpoints."
                    : activeNode === "watch"
                      ? "Compare normalized hidden-state representations across agents and scenarios."
                      : "Bring measured signals, context, and limitations together for investigation."}
                </p>
              </div>
            )}
          </div>
        </div>
      </section>

      <section className="fit-section">
        <div className="section-heading">
          <div>
            <div className="eyebrow">WHERE IT FITS</div>
            <h2>Internal observability for AI infrastructure.</h2>
          </div>
          <p>
            MechGuard complements the telemetry already surrounding modern AI
            systems.
          </p>
        </div>

        <div className="fit-grid">
          {[
            ["AI platforms", "Model serving, evaluation and deployment stacks.", BrainCircuit],
            ["AI safety", "Mechanistic evidence below the input/output boundary.", ShieldCheck],
            ["Agent infrastructure", "Signals across systems where several agents interact.", Network],
            ["ML infrastructure", "Training checkpoints and model-state instrumentation.", Layers3],
          ].map(([title, copy, Icon], index) => {
            const selected = useCase === index;
            const C = Icon as typeof BrainCircuit;

            return (
              <button
                key={title as string}
                className={`fit-card ${selected ? "is-selected" : ""}`}
                type="button"
                onClick={() => setUseCase(selected ? null : index)}
              >
                <C size={18} />
                <strong>{title as string}</strong>
                <span>{copy as string}</span>
              </button>
            );
          })}
        </div>
      </section>

      <section className="layer-section">
        <div className="section-heading">
          <div>
            <div className="eyebrow">WHY THIS LAYER</div>
            <h2>Existing telemetry sees the boundary.</h2>
          </div>
          <p>
            MechGuard looks one layer deeper, where internal model signals can
            change before output behavior makes the change obvious.
          </p>
        </div>

        <div className="comparison-shell">
          {[
            [
              "Traditional telemetry",
              "Requests, outputs, latency, errors and infrastructure health.",
            ],
            [
              "MechGuard",
              "Weight-space geometry and hidden-state representations.",
            ],
            [
              "Together",
              "Boundary telemetry plus internal model-state evidence.",
            ],
          ].map(([title, copy], index) => (
            <button
              key={title}
              className={`comparison-card ${
                comparison === index ? "is-selected" : ""
              }`}
              type="button"
              onClick={() => setComparison(comparison === index ? null : index)}
            >
              <span className="comparison-number">0{index + 1}</span>
              <strong>{title}</strong>
              <p>{copy}</p>
            </button>
          ))}
        </div>
      </section>

      <section className="workspace-section" ref={workspaceRef} id="workspace">
        <div className="workspace-heading">
          <div>
            <div className="eyebrow">INTERACTIVE PRODUCT WORKSPACE</div>
            <h2>Don’t just read about it. Run the evidence.</h2>
            <p>
              Select a monitor, provide a payload, and send it to the MechGuard
              analysis endpoint.
            </p>
          </div>

          <div className="workspace-status">
            <span className="status-dot" />
            LIVE PROTOTYPE
          </div>
        </div>

        <div className="workspace-shell">
          <aside className="workspace-sidebar">
            <div className="sidebar-label">MONITORS</div>

            {layers.map((layer) => {
              const Icon = layer.icon;

              return (
                <button
                  key={layer.id}
                  className={`workspace-tab ${
                    stage === layer.id ? "is-selected" : ""
                  }`}
                  type="button"
                  onClick={() => {
                    setStage(layer.id);
                    setResult(null);
                    setError("");
                  }}
                >
                  <Icon size={16} />
                  <span>
                    <strong>{layer.title}</strong>
                    <small>{layer.eyebrow}</small>
                  </span>
                  <ArrowRight size={14} />
                </button>
              );
            })}

            <div className="sidebar-note">
              <Activity size={14} />
              <span>
                Research-backed prototype. Results are returned by the current
                backend rather than simulated in the UI.
              </span>
            </div>
          </aside>

          <div className="workspace-main">
            {stage === null ? (
              <div className="workspace-empty">
                <div className="empty-icon">
                  <Network size={22} />
                </div>
                <div className="eyebrow">NO MONITOR SELECTED</div>
                <h3>Choose a monitor</h3>
                <p>Select Attest, Watch or Review from the left.</p>
              </div>
            ) : (
              <>
                <div className="workspace-toolbar">
                  <div>
                    <div className="eyebrow">
                      {layers.find((item) => item.id === stage)?.eyebrow}
                    </div>
                    <h3>
                      {layers.find((item) => item.id === stage)?.title}
                    </h3>
                  </div>

                  <button
                    className="icon-button"
                    onClick={clearInput}
                    type="button"
                    title="Clear"
                  >
                    <RefreshCw size={15} />
                  </button>
                </div>

                <div
                  className="dropzone"
                  onDragOver={(event) => event.preventDefault()}
                  onDrop={onDrop}
                >
                  <input
                    id="mechguard-file"
                    type="file"
                    hidden
                    onChange={(event) => chooseFile(event.target.files?.[0])}
                  />

                  <div className="dropzone-icon">
                    <CloudUpload size={21} />
                  </div>

                  <strong>
                    {fileName || "Drop a payload here"}
                  </strong>
                  <span>
                    or paste your model input below
                  </span>

                  <label htmlFor="mechguard-file" className="secondary-button">
                    <Upload size={14} />
                    Choose file
                  </label>
                </div>

                <textarea
                  className="payload-input"
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  placeholder="Paste JSON, text, or an analysis payload..."
                  spellCheck={false}
                />

                {error && <div className="workspace-error">{error}</div>}

                <div className="workspace-actions">
                  <button
                    className="primary-button"
                    type="button"
                    onClick={runAnalysis}
                    disabled={running}
                  >
                    {running ? (
                      <>
                        <RefreshCw className="spin" size={15} />
                        Running…
                      </>
                    ) : (
                      <>
                        <Play size={15} />
                        Run analysis
                      </>
                    )}
                  </button>

                  {input && (
                    <button
                      className="text-link"
                      type="button"
                      onClick={clearInput}
                    >
                      <X size={14} />
                      Clear
                    </button>
                  )}
                </div>

                {result && (
                  <div className="result-panel">
                    <div className="result-header">
                      <div>
                        <div className="eyebrow">ANALYSIS RESULT</div>
                        <h3>Backend response</h3>
                      </div>
                      <span className="result-live">RETURNED</span>
                    </div>

                    <pre>{JSON.stringify(result, null, 2)}</pre>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </section>

      <section className="evidence-section">
        <div className="section-heading">
          <div>
            <div className="eyebrow">RESEARCH EVIDENCE</div>
            <h2>Built from measured signals.</h2>
          </div>
          <p>
            Current evidence is exploratory and explicitly separated from
            claims that remain unvalidated.
          </p>
        </div>

        <div className="evidence-grid">
          {evidence.map((item) => (
            <button
              key={item.label}
              className="evidence-card"
              type="button"
              onClick={() => scrollToWorkspace(item.target)}
            >
              <div className="evidence-top">
                <span>{item.label}</span>
                <ArrowRight size={15} />
              </div>

              <h3>{item.title}</h3>
              <p>{item.description}</p>

              <div className="evidence-metric">
                <strong>{item.metric}</strong>
                <span>{item.metricLabel}</span>
              </div>
            </button>
          ))}
        </div>
      </section>

      <section className="final-cta">
        <div>
          <div className="eyebrow">MECHGUARD</div>
          <h2>See what changes underneath.</h2>
          <p>
            Internal observability for the next generation of AI systems.
          </p>
        </div>

        <button
          className="primary-button"
          type="button"
          onClick={() => scrollToWorkspace()}
        >
          Open workspace
          <ArrowRight size={16} />
        </button>
      </section>

      <footer className="site-footer">
        <span>MECHGUARD</span>
        <span>Internal AI observability</span>
        <span>EdgeDaemon</span>
      </footer>
    </main>
  );
}
