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

    root.style.removeProperty("background-color");
    root.style.removeProperty("color");
    document.body.style.removeProperty("background-color");
    document.body.style.removeProperty("color");
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
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    localStorage.setItem("mechguard-theme", next);
    applyTheme(next);
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
            <span style={{ display: "inline-flex", alignItems: "center", gap: "6px" }}>
              <span className="pulse-indicator" /> AGENTIC ORCHESTRATION MESH
            </span>
            <span>LIVE SIGNAL PROBES</span>
          </div>

          <svg
            className="signal-svg"
            viewBox="0 0 620 340"
            role="img"
            aria-label="Agentic orchestration and internal signal flow visualization"
          >
            <defs>
              <filter id="glow" x="-30%" y="-30%" width="160%" height="160%">
                <feGaussianBlur stdDeviation="3" result="blur" />
                <feComposite in="SourceGraphic" in2="blur" operator="over" />
              </filter>
            </defs>

            {/* Background Grid Pattern */}
            <g stroke="currentColor" strokeOpacity="0.06" strokeWidth="0.8">
              <line x1="90" y1="0" x2="90" y2="340" />
              <line x1="270" y1="0" x2="270" y2="340" />
              <line x1="450" y1="0" x2="450" y2="340" />
              <line x1="0" y1="100" x2="620" y2="100" />
              <line x1="0" y1="240" x2="620" y2="240" />
            </g>

            {/* CONNECTING PATHWAYS */}
            {/* Path 1: Primary Agent (90, 170) -> Orchestrator Diamond (270, 100) */}
            <path
              id="flowPrimaryToOrchestrator"
              d="M 90 170 C 140 170, 200 100, 270 100"
              fill="none"
              stroke="currentColor"
              strokeOpacity="0.22"
              strokeWidth="1.5"
              strokeDasharray="4 4"
            />

            {/* Path 2: Orchestrator (270, 100) -> Sub-Agent A (450, 70) */}
            <path
              id="flowOrchestratorToAgent1"
              d="M 270 100 C 330 100, 390 70, 450 70"
              fill="none"
              stroke="currentColor"
              strokeOpacity="0.22"
              strokeWidth="1.5"
            />

            {/* Path 3: Sub-Agent A (450, 70) -> Sub-Agent B (450, 240) */}
            <path
              id="flowAgent1ToAgent2"
              d="M 450 70 C 490 155, 490 155, 450 240"
              fill="none"
              stroke="currentColor"
              strokeOpacity="0.2"
              strokeWidth="1.5"
              strokeDasharray="3 3"
            />

            {/* Path 4: Orchestrator (270, 100) -> MechGuard Probe (270, 240) */}
            <path
              id="flowToProbe"
              d="M 270 100 L 270 240"
              fill="none"
              stroke="var(--accent)"
              strokeOpacity="0.45"
              strokeWidth="2"
            />

            {/* Path 5: MechGuard Probe (270, 240) -> Hidden State Telemetry (540, 240) */}
            <path
              id="flowProbeToOutput"
              d="M 270 240 Q 405 240, 540 240"
              fill="none"
              stroke="currentColor"
              strokeOpacity="0.2"
              strokeWidth="1.5"
            />

            {/* Path 6: Loopback Sub-Agent B (450, 240) -> Primary Agent (90, 170) */}
            <path
              id="flowLoopback"
              d="M 450 240 C 350 310, 150 270, 90 170"
              fill="none"
              stroke="currentColor"
              strokeOpacity="0.15"
              strokeWidth="1"
              strokeDasharray="6 4"
            />

            {/* ANIMATED GLOWING PULSES ALONG PATHWAYS */}
            <circle r="4" fill="var(--accent)" filter="url(#glow)">
              <animateMotion dur="3.2s" repeatCount="indefinite">
                <mpath href="#flowPrimaryToOrchestrator" />
              </animateMotion>
            </circle>

            <circle r="3.5" fill="var(--accent)" filter="url(#glow)">
              <animateMotion dur="2.4s" repeatCount="indefinite">
                <mpath href="#flowOrchestratorToAgent1" />
              </animateMotion>
            </circle>

            <circle r="3" fill="var(--accent)">
              <animateMotion dur="3.8s" repeatCount="indefinite">
                <mpath href="#flowAgent1ToAgent2" />
              </animateMotion>
            </circle>

            <circle r="4.5" fill="var(--accent)" filter="url(#glow)">
              <animateMotion dur="1.8s" repeatCount="indefinite">
                <mpath href="#flowToProbe" />
              </animateMotion>
            </circle>

            <circle r="3.5" fill="var(--accent)" opacity="0.8" filter="url(#glow)">
              <animateMotion dur="4.0s" repeatCount="indefinite">
                <mpath href="#flowProbeToOutput" />
              </animateMotion>
            </circle>

            <circle r="2.5" fill="var(--accent)" opacity="0.7">
              <animateMotion dur="5.5s" repeatCount="indefinite">
                <mpath href="#flowLoopback" />
              </animateMotion>
            </circle>

            {/* NODE 1: PRIMARY AGENT (Circle Node at 90, 170) */}
            <g transform="translate(90, 170)">
              {/* Radar ring animation */}
              <circle r="28" fill="none" stroke="currentColor" strokeOpacity="0.2" strokeWidth="1">
                <animate attributeName="r" values="22;34;22" dur="4s" repeatCount="indefinite" />
                <animate attributeName="stroke-opacity" values="0.3;0.05;0.3" dur="4s" repeatCount="indefinite" />
              </circle>
              {/* Outer boundary */}
              <circle r="22" fill="var(--surface)" stroke="currentColor" strokeOpacity="0.4" strokeWidth="1.5" />
              {/* Inner core */}
              <circle r="12" fill="var(--accent-soft)" stroke="var(--accent)" strokeWidth="1.5" />
              <circle r="5" fill="var(--accent)" filter="url(#glow)">
                <animate attributeName="opacity" values="0.4;1;0.4" dur="2s" repeatCount="indefinite" />
              </circle>
              <text y="38" textAnchor="middle" fill="currentColor" fontSize="9" fontWeight="700" opacity="0.85" letterSpacing="0.08em">
                PRIMARY MODEL
              </text>
            </g>

            {/* NODE 2: AGENT ORCHESTRATOR (Diamond Node at 270, 100) */}
            <g transform="translate(270, 100)">
              {/* Outer diamond outline */}
              <polygon
                points="0,-32 32,0 0,32 -32,0"
                fill="none"
                stroke="var(--accent)"
                strokeOpacity="0.3"
                strokeWidth="1"
                strokeDasharray="4 4"
              />
              {/* Main Diamond */}
              <polygon
                points="0,-24 24,0 0,24 -24,0"
                fill="var(--surface)"
                stroke="var(--accent)"
                strokeWidth="1.5"
              />
              {/* Inner Diamond Core */}
              <polygon points="0,-10 10,0 0,10 -10,0" fill="var(--accent)" filter="url(#glow)">
                <animate attributeName="opacity" values="0.6;1;0.6" dur="2.5s" repeatCount="indefinite" />
              </polygon>
              <text y="44" textAnchor="middle" fill="currentColor" fontSize="9" fontWeight="700" opacity="0.85" letterSpacing="0.08em">
                ORCHESTRATOR
              </text>
            </g>

            {/* NODE 3: SUB-AGENT A (Hexagon Node at 450, 70) */}
            <g transform="translate(450, 70)">
              <polygon
                points="0,-18 16,-9 16,9 0,18 -16,9 -16,-9"
                fill="var(--surface)"
                stroke="currentColor"
                strokeOpacity="0.45"
                strokeWidth="1.5"
              />
              <circle r="4" fill="var(--accent)" />
              <text y="32" textAnchor="middle" fill="currentColor" fontSize="8" fontWeight="600" opacity="0.75" letterSpacing="0.06em">
                AGENT A
              </text>
            </g>

            {/* NODE 4: SUB-AGENT B (Hexagon Node at 450, 240) */}
            <g transform="translate(450, 240)">
              <polygon
                points="0,-18 16,-9 16,9 0,18 -16,9 -16,-9"
                fill="var(--surface)"
                stroke="currentColor"
                strokeOpacity="0.45"
                strokeWidth="1.5"
              />
              <circle r="4" fill="var(--accent)" />
              <text y="32" textAnchor="middle" fill="currentColor" fontSize="8" fontWeight="600" opacity="0.75" letterSpacing="0.06em">
                AGENT B
              </text>
            </g>

            {/* NODE 5: MECHGUARD OBSERVABILITY PROBE (Shield/Concentric Node at 270, 240) */}
            <g transform="translate(270, 240)">
              <circle r="26" fill="none" stroke="var(--accent)" strokeOpacity="0.25" strokeWidth="1" />
              <circle r="18" fill="var(--surface)" stroke="var(--accent)" strokeWidth="2" />
              {/* Inner Shield Crosshair */}
              <path d="M -8 0 L 8 0 M 0 -8 L 0 8" stroke="var(--accent)" strokeWidth="1.5" />
              <circle r="3" fill="var(--accent)" filter="url(#glow)" />
              <text y="40" textAnchor="middle" fill="var(--accent)" fontSize="9" fontWeight="700" letterSpacing="0.08em">
                MECHGUARD PROBE
              </text>
            </g>

            {/* NODE 6: LIVE ACTIVATION / WAVE MONITOR (at 540, 240) */}
            <g transform="translate(540, 240)">
              <rect
                x="-30"
                y="-20"
                width="60"
                height="40"
                rx="8"
                fill="var(--surface-2)"
                stroke="currentColor"
                strokeOpacity="0.2"
                strokeWidth="1"
              />
              {/* Animated Wave signal inside monitor */}
              <path
                d="M -22 0 Q -15 -12 -8 0 T 6 0 T 20 0"
                fill="none"
                stroke="var(--accent)"
                strokeWidth="1.5"
              >
                <animate
                  attributeName="d"
                  values="M -22 0 Q -15 -12 -8 0 T 6 0 T 20 0; M -22 0 Q -15 12 -8 0 T 6 0 T 20 0; M -22 0 Q -15 -12 -8 0 T 6 0 T 20 0"
                  dur="2s"
                  repeatCount="indefinite"
                />
              </path>
              <text y="32" textAnchor="middle" fill="currentColor" fontSize="8" fontWeight="600" opacity="0.6">
                PROBE SIGNALS
              </text>
            </g>
          </svg>

          <div className="hero-node-labels">
            <span>MULTI-AGENT ORCHESTRATION</span>
            <span>WEIGHT-SPACE & HIDDEN STATES</span>
            <span>REAL-TIME TELEMETRY</span>
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

        <div className="product-grid-shell">
          <div className="product-pipeline-bar">
            <div className="product-pipeline-step">
              <ShieldCheck size={16} />
              <span>01 ATTEST</span>
            </div>
            <div className="product-pipeline-connector">
              <div className="product-pipeline-pulse" />
            </div>

            <div className="product-pipeline-step">
              <Eye size={16} />
              <span>02 WATCH</span>
            </div>
            <div className="product-pipeline-connector">
              <div className="product-pipeline-pulse" />
            </div>

            <div className="product-pipeline-step">
              <Activity size={16} />
              <span>03 REVIEW</span>
            </div>
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
                    <span>STAGE 0{index + 1}</span>
                    <Icon size={18} />
                  </div>
                  <div>
                    <span className="product-card-badge">{layer.eyebrow}</span>
                    <h3>{layer.title}</h3>
                    <p>{layer.description}</p>

                    <div className="product-card-stream">
                      <span>LIVE TELEMETRY</span>
                      <strong>
                        {layer.id === "attest"
                          ? "8.36× SVD Growth"
                          : layer.id === "watch"
                            ? "0.9677 AUROC"
                            : "A001 / B003 Artifacts"}
                      </strong>
                    </div>
                  </div>
                  <ArrowRight className="card-arrow" size={16} />
                </button>
              );
            })}
          </div>
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
              ["instrument", "Instrument", Layers3, "attest"],
              ["measure", "Measure", Activity, null],
              ["surface", "Surface", Network, "watch"],
              ["review", "Review", FileCode2, "review"],
            ].map(([id, title, Icon, targetStage], index) => {
              const selected =
                activeNode === targetStage || (id === "measure" && activeNode === null);

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
                    <div className="node-header">
                      <span className="node-index">STEP 0{index + 1}</span>
                      <span className="node-status-dot" />
                    </div>
                    <C size={20} />
                    <strong>{title as string}</strong>
                  </button>

                  {index < 3 && <ArrowRight className={`pipeline-arrow ${selected ? "active" : ""}`} size={16} />}
                </div>
              );
            })}
          </div>

          <div className="architecture-detail">
            {activeNode === null ? (
              <div>
                <div className="eyebrow">SELECTED SIGNAL — STAGE 02</div>
                <h3>Measure Weight-Space & Activation Trajectories</h3>
                <p>
                  Perform SVD singular value decomposition, delta-W subspace alignment checks, and representation commutator metrics directly across model hidden layers.
                </p>
                <div className="architecture-metrics-grid">
                  <div className="architecture-metric-item">
                    <small>TOP SINGULAR VALUE GROWTH</small>
                    <span>8.36× (A001 Run)</span>
                  </div>
                  <div className="architecture-metric-item">
                    <small>LAYER COVERAGE</small>
                    <span>Layers 26–30 (Qwen3-32B)</span>
                  </div>
                  <div className="architecture-metric-item">
                    <small>METRIC TYPE</small>
                    <span>{"Principal Angles \\(\\theta_{max}\\)"}</span>
                  </div>
                </div>
              </div>
            ) : (
              <div>
                <div className="eyebrow">SELECTED SIGNAL — {activeNode.toUpperCase()} STAGE</div>
                <h3>
                  {activeNode === "attest"
                    ? "Training-time internal change attestation"
                    : activeNode === "watch"
                      ? "Cross-agent hidden state representation monitoring"
                      : "Evidence bundle for engineering review"}
                </h3>
                <p>
                  {activeNode === "attest"
                    ? "Instruments weight-space geometry evolution across fine-tuning checkpoints to attest model stability before deployment."
                    : activeNode === "watch"
                      ? "Monitors multi-agent hidden representations in deployment, identifying representation drift and protocol divergence."
                      : "Aggregates measured internal signals, execution manifests, and limitations into actionable engineering diagnostic reports."}
                </p>
                <div className="architecture-metrics-grid">
                  <div className="architecture-metric-item">
                    <small>TARGET STAGE</small>
                    <span>{activeNode === "attest" ? "Fine-tuning / SFT" : activeNode === "watch" ? "Inference & Multi-Agent Mesh" : "Engineering & Audit"}</span>
                  </div>
                  <div className="architecture-metric-item">
                    <small>EVIDENCE ASSET</small>
                    <span>{activeNode === "attest" ? "A001 Archive" : activeNode === "watch" ? "B001 / B003 Manifest" : "Full Evidence Report"}</span>
                  </div>
                </div>
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
            Click any infrastructure domain below to inspect MechGuard's capabilities.
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
                <C size={20} />
                <strong>{title as string}</strong>
                <span>{copy as string}</span>
              </button>
            );
          })}
        </div>

        {useCase !== null && (
          <div className="fit-drawer">
            <div className="fit-drawer-header">
              <div className="eyebrow">SYSTEM CAPABILITY EXPANSION</div>
              <button
                className="icon-button"
                onClick={() => setUseCase(null)}
                type="button"
                title="Close"
              >
                <X size={15} />
              </button>
            </div>

            <h3>
              {useCase === 0
                ? "AI Platforms — Hidden-State Proxy Hooks"
                : useCase === 1
                  ? "AI Safety — Mechanistic Representation Analysis"
                  : useCase === 2
                    ? "Agent Infrastructure — Multi-Agent Protocol Alignment"
                    : "ML Infrastructure — Checkpoint Geometry Attestation"}
            </h3>

            <p>
              {useCase === 0
                ? "Integrates into inference engines (vLLM, Ollama, TensorRT-LLM) to perform real-time hidden-state representation monitoring. Catches internal representation drift before it yields hallucinated outputs or safety violations."
                : useCase === 1
                  ? "Bypasses superficial prompt/completion filtering by measuring internal neural geometry directly. Evaluates weight-space delta shifts and representation alignment to uncover hidden model shifts during fine-tuning."
                  : useCase === 2
                    ? "Monitors multi-agent systems where several LLMs interact. Evaluates hidden-state representation overlap across agent communication chains, flagging state misalignment and out-of-distribution message patterns."
                    : "Instruments model fine-tuning runs across multiple checkpoints. Analyzes singular value trajectories, weight-matrix norms, and subspace stability across LoRA / QLoRA / SFT training runs."}
            </p>

            <div className="fit-drawer-tags">
              {useCase === 0 && (
                <>
                  <span className="fit-drawer-tag"><Zap size={12} /> Inference Proxy Integration</span>
                  <span className="fit-drawer-tag"><Activity size={12} /> Real-Time Representation Drift</span>
                  <span className="fit-drawer-tag"><ShieldCheck size={12} /> Zero Latency Impact</span>
                </>
              )}
              {useCase === 1 && (
                <>
                  <span className="fit-drawer-tag"><ShieldCheck size={12} /> Mechanistic Safety Probes</span>
                  <span className="fit-drawer-tag"><Layers3 size={12} /> Commutator Matrix Analysis</span>
                  <span className="fit-drawer-tag"><BrainCircuit size={12} /> Jailbreak Subspace Tracking</span>
                </>
              )}
              {useCase === 2 && (
                <>
                  <span className="fit-drawer-tag"><Network size={12} /> Inter-Agent Protocol Safety</span>
                  <span className="fit-drawer-tag"><Activity size={12} /> Scenario AUROC: 0.9677</span>
                  <span className="fit-drawer-tag"><Eye size={12} /> Multi-Agent State Alignment</span>
                </>
              )}
              {useCase === 3 && (
                <>
                  <span className="fit-drawer-tag"><Layers3 size={12} /> Checkpoint SVD Trajectories</span>
                  <span className="fit-drawer-tag"><Zap size={12} /> LoRA Rank & Alpha Bounds</span>
                  <span className="fit-drawer-tag"><Activity size={12} /> Weight Subspace Attestation</span>
                </>
              )}
            </div>
          </div>
        )}
      </section>

      <section className="layer-section">
        <div className="section-heading">
          <div>
            <div className="eyebrow">WHY THIS LAYER</div>
            <h2>Existing telemetry sees the boundary.</h2>
          </div>
          <p>
            Click a card below to explore observability depth comparison.
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

        {comparison !== null && (
          <div className="fit-drawer">
            <div className="fit-drawer-header">
              <div className="eyebrow">OBSERVABILITY DEPTH MATRIX</div>
              <button
                className="icon-button"
                onClick={() => setComparison(null)}
                type="button"
                title="Close"
              >
                <X size={15} />
              </button>
            </div>

            <h3>
              {comparison === 0
                ? "Perimeter & Infrastructure Observability (Traditional)"
                : comparison === 1
                  ? "Deep Model-State & Weight Geometry (MechGuard)"
                  : "360° Complete Observability Stack (Together)"}
            </h3>

            <p>
              {comparison === 0
                ? "Traditional APMs and LLM logging tools monitor API status codes, request throughput, token latency (p95/p99), and text outputs. While vital for operations, they remain blind to internal model degradation or subtle weight shifts."
                : comparison === 1
                  ? "MechGuard probes inside the model — analyzing singular value decomposition, hidden-state representation alignment, and layer-by-layer principal angles. It detects model shift before surface behavior breaks."
                  : "By coupling traditional perimeter telemetry with MechGuard's internal mechanistic probes, platform teams gain 360° visibility — tracking everything from HTTP API requests down to neural network hidden states."}
            </p>

            <div className="fit-drawer-tags">
              <span className="fit-drawer-tag"><Activity size={12} /> Scope: {comparison === 0 ? "Boundary Only" : comparison === 1 ? "Neural Layers Only" : "Perimeter + Deep Neural States"}</span>
              <span className="fit-drawer-tag"><ShieldCheck size={12} /> Early Warning: {comparison === 0 ? "Post-Output Failure" : "Pre-Boundary Signal Shift"}</span>
            </div>
          </div>
        )}
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
