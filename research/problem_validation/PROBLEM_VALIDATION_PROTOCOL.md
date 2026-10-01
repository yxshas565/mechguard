# MechGuard — Problem Validation Protocol

## Status

Problem Validation has started following completion of Phase 1C.

This phase validates the real-world problem surrounding AI model transformation,
safety evidence reuse, revalidation, and deployment decisions.

The purpose is to determine whether the problem is experienced by real
practitioners, how it is currently handled, where the operational pain occurs,
and whether an unmet need exists.

This phase does not assume that MechGuard is the required solution.

---

## 1. Current Problem Hypothesis

When an AI model is transformed through fine-tuning, LoRA adaptation,
merging, quantization, distillation, editing, or related modification,
teams may need to determine whether safety and evaluation evidence collected
for the original model remains applicable to the transformed model.

The hypothesis to validate is that this process can create a meaningful
operational burden because teams must decide:

1. Which previous evaluations remain valid.
2. Which evaluations must be rerun.
3. What evidence is sufficient to justify deployment.
4. How internal model changes affect confidence in previous evidence.
5. How to document the transformation and resulting validation decision.

The existence, severity, frequency, and economic importance of this problem
remain hypotheses until supported by practitioner evidence.

---

## 2. Primary User Groups

Initial discovery will investigate practitioners involved in:

1. ML engineering
2. LLM fine-tuning and adaptation
3. AI evaluation
4. AI safety and security
5. Model governance and risk
6. ML infrastructure and deployment
7. Technical compliance and audit

These are discovery categories, not confirmed customer personas.

---

## 3. Core Research Questions

### RQ1 — Transformation Workflow

How do teams currently modify and adapt foundation models?

Investigate:

1. Fine-tuning
2. LoRA or PEFT adaptation
3. Model merging
4. Quantization
5. Distillation
6. Model editing
7. Other production transformations

### RQ2 — Evaluation Workflow

What evaluations are performed after a model transformation?

Investigate:

1. Which evaluations are automatically rerun.
2. Which evaluations are selectively rerun.
3. Which evaluations are skipped.
4. What determines the decision.
5. Who owns the decision.

### RQ3 — Evidence Reuse

How do teams decide whether evidence from the original model
still applies after transformation?

Investigate:

1. Existing benchmark results.
2. Safety evaluations.
3. Red-team results.
4. Internal testing.
5. Model lineage.
6. Transformation metadata.
7. Human review.
8. Informal engineering judgment.

### RQ4 — Pain

Where does the current workflow create friction?

Investigate:

1. Time
2. Compute
3. Engineering effort
4. Evaluation cost
5. Manual work
6. Uncertainty
7. Documentation burden
8. Review delays
9. Deployment delays
10. Risk of insufficient validation

### RQ5 — Failure Modes

Have teams encountered situations where a transformed model behaved
differently from the original model despite previous evidence appearing
acceptable?

Capture:

1. What transformation occurred.
2. What evidence existed beforehand.
3. What changed.
4. How the change was discovered.
5. What remediation occurred.
6. What additional evaluation was required.

Do not assume that such incidents exist.

### RQ6 — Existing Solutions

What tools, processes, or internal systems currently address this problem?

Investigate:

1. Evaluation frameworks
2. Experiment tracking
3. Model registries
4. Safety benchmark suites
5. Red-team tooling
6. Governance workflows
7. Internal scripts
8. Manual review
9. Other processes

### RQ7 — Importance

If the problem exists, how important is it relative to other engineering
and safety priorities?

Investigate:

1. Frequency
2. Severity
3. Cost
4. Deployment impact
5. Organizational ownership
6. Existing budget
7. Current workaround
8. Willingness to change workflow

---

## 4. Interview Rules

The interviewer must not begin by explaining MechGuard.

Do not ask:

1. "Would you use MechGuard?"
2. "Would you pay for this?"
3. "Do you think this product is useful?"
4. "Would internal weight monitoring solve this?"

Instead ask about actual past and current behavior.

Prefer:

1. "Walk me through the last time you fine-tuned or adapted a model."
2. "What did you evaluate before deployment?"
3. "Which existing evaluations did you reuse?"
4. "Which ones did you rerun?"
5. "How did you decide?"
6. "What was the most painful part?"
7. "How long did that take?"
8. "What tools did you use?"
9. "Has this ever caused a deployment delay?"
10. "Have you ever discovered a behavioral change after transformation?"

Past behavior is stronger evidence than hypothetical enthusiasm.

---

## 5. Evidence Classification

Every interview observation should be classified as one of:

### Direct Evidence

The participant describes an actual workflow, incident, cost,
tool, decision, or repeated behavior.

### Reported Concern

The participant believes something could become a problem but has
not personally experienced it.

### Hypothetical Interest

The participant expresses interest in a proposed capability without
describing an existing problem or workflow.

### Contradictory Evidence

The participant reports that the suspected problem does not materially
affect their workflow.

Hypothetical interest must not be treated as proof of problem validation.

---

## 6. Interview Record

Each interview should capture:

| Field | Description |
|---|---|
| Participant role | Role relevant to model lifecycle |
| Organization type | Startup, enterprise, research, etc. |
| Model workflow | How models are adapted/deployed |
| Transformation types | Transformations actually used |
| Evaluation process | What is evaluated and when |
| Evidence reuse | How prior evidence is reused |
| Pain point | Specific observed friction |
| Frequency | How often it occurs |
| Cost | Time, compute, people, or money |
| Existing workaround | Current solution |
| Incident | Actual failure or regression if any |
| Decision owner | Who decides deployment readiness |
| Severity | Consequence of the problem |
| Evidence class | Direct / reported / hypothetical / contradictory |
| Quote | Short verbatim evidence where permission allows |

---

## 7. Problem Validation Thresholds

No threshold is considered satisfied merely because someone likes the
MechGuard concept.

Evidence should accumulate across independent participants.

The following dimensions will be assessed:

1. Problem occurrence
2. Problem frequency
3. Operational pain
4. Resource cost
5. Existing workaround weakness
6. Consequence of failure
7. Clear ownership
8. Existing budget or evaluation effort
9. Repeated evidence across organizations

The thresholds themselves are hypotheses and may be revised based on
discovery evidence.

---

## 8. Decision Outcomes

At the end of discovery, the problem hypothesis may be classified as:

### Validated

Strong repeated evidence of a meaningful existing problem.

### Partially Validated

The problem exists, but only for particular workflows, organizations,
transformations, or use cases.

### Weakly Supported

The problem is recognized but evidence of meaningful operational pain
is limited.

### Not Validated

Practitioner evidence does not support the proposed problem as a
meaningful current pain point.

### Reframed

Discovery reveals a different and more important problem than the
original hypothesis.

A negative or reframed result is considered a valid research outcome.

---

## 9. Solution Separation

Problem validation must finish before solution validation.

The sequence is:

Problem
→ Workflow
→ Pain
→ Existing workaround
→ Unmet need
→ Solution hypothesis
→ Solution validation

MechGuard should only be positioned as the solution after the underlying
problem has sufficient evidence.

---

## 10. Research Artifacts

This phase will produce:

1. `PROBLEM_VALIDATION_PROTOCOL.md`
2. `PAIN_POINT_HYPOTHESES.md`
3. `INTERVIEW_GUIDE.md`
4. `INTERVIEW_LOG.md`
5. `WORKFLOW_MAP.md`
6. `EVIDENCE_MATRIX.md`
7. `PROBLEM_VALIDATION_REPORT.md`

The final report must distinguish direct practitioner evidence from
literature evidence, inference, and product hypotheses.

---

## 11. Current Status

Phase 1C technical validation: COMPLETE.

Problem validation protocol: INITIALIZED.

Practitioner discovery: NOT YET STARTED.

Pain points: NOT YET VALIDATED.

Customer/problem fit: NOT YET VALIDATED.

Solution validation: NOT STARTED.
