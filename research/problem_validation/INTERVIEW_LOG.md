# MechGuard — Practitioner Interview Log

## Purpose

This document records practitioner interviews conducted during Problem
Validation.

It is intentionally structured so that direct evidence can be compared
across participants.

Do not record invented details.

Do not convert participant opinions into facts.

Do not remove contradictory evidence.

---

# Interview Index

| ID | Date | Participant Role | Organization Type | Relevant Transformation | Evidence Strength | Status |
|---|---|---|---|---|---|---|
| PV-001 | | | | | | |
| PV-002 | | | | | | |
| PV-003 | | | | | | |
| PV-004 | | | | | | |
| PV-005 | | | | | | |
| PV-006 | | | | | | |
| PV-007 | | | | | | |
| PV-008 | | | | | | |
| PV-009 | | | | | | |
| PV-010 | | | | | | |

---

# Interview Record Template

Copy this section for every completed interview.

---

## PV-XXX

### 1. Participant

Role:

Organization type:

Relevant experience:

Date:

Interview format:

Permission to record:

---

### 2. Model Workflow

Models worked with:

How models are adapted:

Transformations actually used:

Frequency of transformation:

---

### 3. Recent Transformation

Most recent relevant transformation:

Starting model:

Reason for transformation:

Transformation method:

Resulting model:

---

### 4. Evaluation Before Transformation

Existing evidence:

Safety evaluation:

Security evaluation:

Capability evaluation:

Red-team evaluation:

Other evidence:

---

### 5. Evaluation After Transformation

Evaluations performed:

Evaluations skipped:

Reason for selection:

Complete evaluation rerun:

Decision process:

---

### 6. Evidence Reuse

Was previous evidence reused?

If yes, how was applicability determined?

Formal rule:

Internal policy:

Human judgment:

Transformation-specific rule:

Other:

---

### 7. Operational Pain

Primary pain:

Time cost:

Compute cost:

Engineering effort:

Human review:

Deployment impact:

Documentation burden:

Other:

---

### 8. Incident / Unexpected Change

Did the participant describe an actual incident?

Yes / No

If yes:

Transformation:

Previous evidence:

Expected behavior:

Observed behavior:

How discovered:

Impact:

Remediation:

---

### 9. Existing Workaround

Current process:

Tools used:

Automated components:

Manual components:

Known limitations:

---

### 10. Ownership

Decision owner:

Teams involved:

Approval process:

---

### 11. Consequence

What happens when validation is insufficient?

Observed consequence:

Potential consequence:

Frequency:

---

### 12. Evidence Classification

Classify each important observation.

| Observation | Classification | Evidence |
|---|---|---|
| | Direct Evidence | |
| | Reported Concern | |
| | Hypothetical Interest | |
| | Contradictory Evidence | |

---

### 13. Strongest Evidence

Strongest concrete workflow evidence:

Strongest cost evidence:

Strongest incident evidence:

Strongest evidence of an existing workaround:

Strongest contradictory evidence:

---

### 14. Participant Quotes

Only include short quotes that accurately preserve the participant's
meaning.

Quote 1:

Quote 2:

Quote 3:

---

### 15. Researcher Interpretation

Observed pattern:

Possible implication:

Confidence:

Alternative explanation:

---

### 16. Follow-Up

Follow-up required:

Additional evidence requested:

Potential referral:

---

# Evidence Classification Rules

## Direct Evidence

The participant describes something they actually did, measured, observed,
or experienced.

Examples:

"We rerun our safety suite after every fine-tune."

"We spend approximately two GPU days on the evaluation."

"We discovered a refusal regression after deployment."

---

## Reported Concern

The participant believes something could happen but has not personally
experienced it.

Example:

"We worry that a fine-tune could affect safety behavior."

---

## Hypothetical Interest

The participant reacts positively to a possible future solution.

Example:

"That sounds like something I would use."

This is not evidence that the underlying problem is painful.

---

## Contradictory Evidence

The participant reports that the suspected problem does not materially
affect their workflow.

Example:

"We always rerun the full suite automatically and it takes less than an
hour."

Contradictory evidence must remain in the dataset.

---

# Interview Quality Check

Before considering an interview complete, check:

1. [ ] Actual workflow captured.
2. [ ] Recent transformation discussed.
3. [ ] Evaluation process captured.
4. [ ] Evidence reuse decision investigated.
5. [ ] Pain explored without leading.
6. [ ] Operational cost investigated.
7. [ ] Existing workaround captured.
8. [ ] Decision owner identified.
9. [ ] Contradictory evidence recorded.
10. [ ] Facts separated from researcher interpretation.

---

# Researcher Warning

Do not increase validation confidence simply because multiple participants
say that the proposed MechGuard concept sounds useful.

The strongest evidence is repeated concrete behavior surrounding an existing
problem.

The purpose of this log is to preserve that distinction.
