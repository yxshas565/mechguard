import { NextResponse } from "next/server";

const BASE = process.cwd();
const EVIDENCE = `${BASE}/public/evidence`;

async function readJson(path: string) {
  const fs = await import("fs/promises");
  return JSON.parse(await fs.readFile(path, "utf8"));
}

async function readCsv(path: string) {
  const fs = await import("fs/promises");
  const text = await fs.readFile(path, "utf8");
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return [];
  const headers = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const values = line.split(",");
    return Object.fromEntries(
      headers.map((h, i) => {
        const v = values[i] ?? "";
        const n = Number(v);
        return [h, v !== "" && Number.isFinite(n) ? n : v];
      }),
    );
  });
}

export async function POST(request: Request) {
  try {
    const body = await request.json().catch(() => ({}));
    const stage = body?.stage ?? "attest";

    if (stage === "attest") {
      const [geometry, behavior, monitor] = await Promise.all([
        readCsv(`${EVIDENCE}/a001_geometry_timeseries.csv`),
        readCsv(`${EVIDENCE}/a001_geometry_behavior_trajectory.csv`),
        readJson(`${EVIDENCE}/a001_monitor_results.json`),
      ]);

      return NextResponse.json({
        ok: true,
        source: "A001 persisted research evidence",
        live: false,
        stage: "attest",
        model: "unsloth/Llama-3.2-1B-Instruct",
        layer: "model.layers.8.mlp.down_proj",
        matrix_shape: [2048, 8192],
        checkpoint_count: geometry.length,
        geometry,
        behavior,
        monitor_records: monitor,
        interpretation:
          "A001 observed a large increase in the top singular value across the recorded fine-tuning trajectory. This is exploratory evidence of internal geometric change, not a validated misalignment detector.",
        limitations: [
          "Pilot-scale evidence",
          "No validated EM AUROC/AUPRC",
          "No lead-time validation",
          "No cross-seed validation",
          "No causal claim",
          "Several proposed geometry metrics were not executed",
        ],
      });
    }

    if (stage === "watch") {
      return NextResponse.json({
        ok: true,
        source: "B001/B003 persisted research evidence",
        live: false,
        stage: "watch",
        metrics: {
          agent_oof_auroc: 0.9677,
          peak_auroc: 0.9927,
          mean_auroc: 0.975,
          runs: 62,
          scenarios: 31,
          agents: 248,
          layers: [26, 27, 28, 29, 30],
        },
        interpretation:
          "Activation-based probes separated the evaluated coordination conditions in the recorded benchmark runs. Broad representation-distribution confounds remain unresolved.",
        limitations: [
          "No universal detector demonstrated",
          "No causal mechanism demonstrated",
          "No production deployment validation",
          "No adversarial robustness evaluation",
          "Random 50D controls indicate broad distribution confounds remain possible",
        ],
      });
    }

    return NextResponse.json({
      ok: true,
      source: "MechGuard research evidence",
      live: false,
      stage: "review",
      status: "evidence_review",
      message:
        "Review is currently evidence-backed and does not produce an automated safety verdict.",
    });
  } catch (error) {
    console.error(error);
    return NextResponse.json(
      { ok: false, error: "Unable to load persisted research evidence." },
      { status: 500 },
    );
  }
}
