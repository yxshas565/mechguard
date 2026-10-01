# MechGuard — Pain Point Hypotheses

## Purpose

This document defines the hypotheses that practitioner discovery will test.

These are hypotheses, not established facts.

Each hypothesis must be supported, weakened, or rejected using evidence from
actual practitioner workflows.

---

## H1 — Model transformations create a revalidation decision

### Hypothesis

When teams transform an existing foundation model, they must decide which
previous evaluations and safety evidence remain applicable.

### Evidence to seek

1. Actual model transformation workflow.
2. Evaluations performed before transformation.
3. Evaluations performed after transformation.
4. Explicit or implicit decision about evidence reuse.
5. Person responsible for the decision.

### Disconfirming evidence

Teams routinely rerun a fixed complete evaluation suite after every
transformation with negligible decision-making overhead.

---

## H2 — Teams selectively rerun evaluations

### Hypothesis

Teams do not necessarily rerun every existing evaluation after every
transformation and instead select evaluations based on transformation type,
risk, resources, or engineering judgment.

### Evidence to seek

1. Examples of evaluations that were skipped.
2. Reasons for skipping.
3. Examples of evaluations that were rerun.
4. Rules or heuristics used.
5. Whether the process is automated or manual.

### Disconfirming evidence

Teams consistently execute the same complete evaluation process for every
model transformation.

---

## H3 — Evidence reuse is difficult to justify

### Hypothesis

Teams may have evaluation results for the original model but lack a clear
and standardized method for determining how much of that evidence remains
applicable after transformation.

### Evidence to seek

1. Model lineage records.
2. Evaluation reports.
3. Transformation metadata.
4. Review procedures.
5. Explicit evidence reuse criteria.
6. Cases requiring manual justification.

### Disconfirming evidence

Teams already have a standardized, trusted mechanism that automatically
determines evidence applicability after transformation.

---

## H4 — Revalidation has measurable operational cost

### Hypothesis

Repeated evaluation after model transformation consumes meaningful:

1. Compute
2. Engineering time
3. Evaluation time
4. Human review
5. Deployment time

### Evidence to seek

Whenever possible, capture approximate quantities rather than opinions.

Examples:

1. Hours of engineering effort.
2. GPU hours.
3. Evaluation duration.
4. Number of people involved.
5. Deployment delay.

### Disconfirming evidence

Revalidation is effectively negligible in the participant's workflow.

---

## H5 — The cost of revalidation creates a tradeoff

### Hypothesis

Teams face a practical tradeoff between exhaustive revalidation and
faster model iteration.

### Evidence to seek

1. Examples of evaluation scope being reduced.
2. Examples of deployment being delayed.
3. Examples of checks being prioritized.
4. Resource constraints.
5. Pressure to ship.

### Disconfirming evidence

Teams report no meaningful tradeoff because complete revalidation is
already inexpensive and routine.

---

## H6 — Transformation-specific risk is not always represented in existing workflows

### Hypothesis

Existing evaluation workflows may focus primarily on the resulting model
rather than explicitly reasoning about how the transformation itself affects
the applicability of previous evidence.

### Evidence to seek

1. Whether transformation metadata is recorded.
2. Whether evaluation requirements differ by transformation.
3. Whether model lineage is connected to safety evidence.
4. Whether teams explicitly reason about transformation magnitude.

### Disconfirming evidence

Transformation type and magnitude already determine evaluation requirements
through an established workflow.

---

## H7 — There are identifiable decision owners

### Hypothesis

A specific person or function is accountable for deciding whether a
transformed model has sufficient evidence to deploy.

### Evidence to seek

Possible roles include:

1. ML engineer
2. ML platform engineer
3. AI safety engineer
4. Security engineer
5. Model risk function
6. Technical lead
7. Governance or compliance function

Do not assume which role owns the decision.

### Disconfirming evidence

No identifiable decision owner exists because deployment evidence is fully
automated.

---

## H8 — Existing tools do not fully solve the evidence applicability problem

### Hypothesis

Teams use evaluation frameworks, experiment trackers, model registries,
red-team systems, or internal tooling, but these systems may not fully answer
whether previous safety evidence remains applicable after transformation.

### Evidence to seek

For every existing tool mentioned:

1. What problem does it solve?
2. What does it automate?
3. What remains manual?
4. Where does the workflow break down?
5. What additional tooling is required?

### Disconfirming evidence

Existing tooling already provides the required transformation-aware
evidence decision.

---

## H9 — The problem is concentrated in particular workflows

### Hypothesis

Transformation assurance may not be equally important across all model
workflows.

Potentially relevant contexts include:

1. Production LLM fine-tuning
2. Enterprise model adaptation
3. High-risk applications
4. Frequently updated models
5. Multi-model deployments
6. Regulated environments
7. Large-scale model evaluation

This hypothesis must be narrowed based on evidence.

### Disconfirming evidence

Practitioners across relevant workflows report essentially identical
requirements and pain.

---

## H10 — Actual incidents are more valuable than hypothetical concern

### Hypothesis

The strongest validation signal will come from documented or personally
experienced cases where:

1. A model was transformed.
2. Previous evidence existed.
3. Evaluation scope was changed, reduced, or reused.
4. A meaningful behavioral or safety difference was later discovered.

This is an exploratory hypothesis and does not assume such incidents exist.

### Evidence to seek

Capture:

1. Transformation.
2. Previous evidence.
3. Evaluation decision.
4. Observed change.
5. Discovery mechanism.
6. Impact.
7. Remediation.

---

# Evidence Strength

Evidence should be weighted approximately in this order:

1. Repeated observed workflow across independent practitioners.
2. Specific past incident with concrete details.
3. Measured operational cost.
4. Existing workaround or process.
5. Repeated current pain described independently.
6. General concern.
7. Hypothetical interest in a proposed solution.

The final report must not treat all evidence types as equivalent.

---

# Key Unknowns

The following remain unknown until discovery:

1. Whether the problem occurs frequently enough.
2. Whether it creates meaningful operational pain.
3. Which transformation types create the most difficulty.
4. Which organizations experience the problem most strongly.
5. Who owns the decision.
6. What current tools are used.
7. What existing workflows already solve.
8. Whether the problem is large enough to justify a dedicated product.
9. Whether transformation assurance is the correct framing.
10. Whether MechGuard's proposed technical approach is relevant to the
    validated problem.

---

# Research Principle

The objective is not to prove these hypotheses.

The objective is to find out which hypotheses survive contact with real
practitioner workflows.
