# MechGuard — Practitioner Target Map

## Purpose

This document defines the practitioner groups targeted for Problem
Validation.

The objective is to obtain evidence from people who have direct exposure to
model transformation, evaluation, safety validation, deployment, governance,
or related workflows.

This is a research sampling plan, not a customer list.

---

# 1. Primary Practitioner Groups

## Group A — ML / LLM Engineers

### Why relevant

These practitioners may directly perform model adaptation and evaluation.

### Relevant experience

1. Fine-tuning
2. LoRA / PEFT
3. Model adaptation
4. Evaluation pipelines
5. Model deployment

### Evidence sought

1. Actual transformation workflows.
2. Evaluation workflows.
3. Evaluation selection decisions.
4. Revalidation cost.
5. Engineering workarounds.

### Priority

High.

---

## Group B — AI Evaluation Engineers

### Why relevant

These practitioners may own or operate model evaluation systems.

### Relevant experience

1. Benchmarking
2. Safety evaluation
3. Regression testing
4. Red teaming
5. Automated evaluation

### Evidence sought

1. What gets rerun after transformation.
2. How evaluation scope is determined.
3. Evaluation infrastructure.
4. Manual versus automated work.
5. Known evaluation gaps.

### Priority

High.

---

## Group C — AI Safety / Security Engineers

### Why relevant

These practitioners may deal directly with safety or security regressions
after model changes.

### Relevant experience

1. Safety testing
2. Red teaming
3. Model security
4. Adversarial evaluation
5. Deployment controls

### Evidence sought

1. Safety-specific revalidation.
2. Regression incidents.
3. Risk-based evaluation decisions.
4. Existing safety evidence.
5. Consequences of missed regressions.

### Priority

High.

---

## Group D — ML Platform / Infrastructure Engineers

### Why relevant

These practitioners may own model lifecycle infrastructure and deployment
pipelines.

### Relevant experience

1. Model registries
2. CI/CD
3. Evaluation pipelines
4. Deployment systems
5. Model lineage
6. Monitoring

### Evidence sought

1. Where evaluation gates exist.
2. Automation.
3. Model lineage.
4. Deployment approval.
5. Operational bottlenecks.

### Priority

High.

---

## Group E — Model Risk / AI Governance / Compliance

### Why relevant

These practitioners may determine what evidence is required before
deployment in higher-risk environments.

### Relevant experience

1. Model governance
2. AI risk
3. Compliance
4. Audit
5. Model documentation

### Evidence sought

1. Evidence requirements.
2. Documentation requirements.
3. Approval ownership.
4. Model lineage expectations.
5. Transformation-specific requirements.

### Priority

Medium to High.

---

## Group F — Technical Leads / AI Infrastructure Leads

### Why relevant

Technical leads may have visibility across model development, evaluation,
and deployment decisions.

### Evidence sought

1. Organizational workflow.
2. Resource allocation.
3. Deployment tradeoffs.
4. Ownership.
5. Existing tooling.
6. Strategic importance.

### Priority

Medium.

---

# 2. Secondary Practitioner Groups

These groups can provide useful contextual evidence but should not replace
direct practitioners.

## Researchers

Useful for:

1. Understanding emerging workflows.
2. Identifying technical practices.
3. Finding terminology.
4. Discovering relevant incidents.

Research evidence must be distinguished from production practitioner
evidence.

---

## Startup Founders / CTOs

Useful when they personally operate model development and deployment.

Prioritize founders who directly participate in technical workflows rather
than purely business roles.

---

## MLOps Engineers

Useful for:

1. Evaluation pipelines.
2. Model registries.
3. Deployment gates.
4. Reproducibility.
5. Model lineage.

---

# 3. Sampling Strategy

The objective is not to interview ten people with identical backgrounds.

Seek variation across:

1. Role
2. Organization size
3. Industry
4. Model type
5. Transformation type
6. Deployment maturity
7. Evaluation maturity

A useful initial target is:

### Discovery Round

10–15 interviews.

Suggested distribution:

| Group | Initial Target |
|---|---:|
| ML / LLM Engineers | 3 |
| AI Evaluation | 2 |
| AI Safety / Security | 2 |
| ML Platform / Infrastructure | 2 |
| Governance / Risk | 2 |
| Technical Leads / CTOs | 2 |
| Other relevant practitioners | 2 |

These numbers are sampling targets, not validation thresholds.

---

# 4. Strong Candidate Profile

A particularly valuable participant has personally:

1. Modified an existing model.
2. Evaluated the original model.
3. Evaluated the transformed model.
4. Made or influenced the decision about evaluation scope.
5. Experienced the operational consequences of that decision.

The strongest interviews are based on recent concrete workflows.

---

# 5. Low-Value Participants

Do not prioritize participants who:

1. Have never worked with model transformation.
2. Only have theoretical knowledge.
3. Cannot describe an actual workflow.
4. Only react to the MechGuard concept.
5. Are interested solely because the topic sounds novel.

These people may still provide context but should not dominate the evidence.

---

# 6. Candidate Qualification Questions

Before scheduling an interview, establish whether the person has relevant
experience.

Use concise questions such as:

1. "Do you currently work with adapting or fine-tuning existing AI models?"
2. "Have you been involved in evaluating models before or after adaptation?"
3. "Do you work with model deployment or production validation?"
4. "Are you involved in deciding which evaluations need to be rerun?"

A "yes" to one or more relevant questions is sufficient for exploratory
screening.

Do not reveal the hypothesis in the screening message.

---

# 7. Evidence Diversity

Track the following dimensions across interviews.

| Dimension | Target |
|---|---|
| ML engineering | Multiple participants |
| Evaluation | Multiple participants |
| Safety / security | Multiple participants |
| Infrastructure | Multiple participants |
| Governance | Multiple participants |
| Startup | At least some |
| Larger organization | At least some |
| Production models | Majority |
| Different transformation types | Multiple |
| Different evaluation maturity levels | Multiple |

The purpose is to avoid validating the problem only inside one technical
subculture.

---

# 8. Recruitment Sources

Potential sources include:

1. LinkedIn
2. Professional communities
3. Open-source communities
4. AI engineering communities
5. Research communities
6. Existing professional network
7. University / alumni network
8. Founder networks
9. Technical conferences
10. GitHub contributors

Recruitment should target people based on relevant experience rather than
their likelihood of agreeing with the hypothesis.

---

# 9. Outreach Tracking

Every contacted person should eventually be tracked as:

| Candidate | Role | Relevant Experience | Source | Contacted | Response | Interview | Evidence Quality |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

Do not treat a response or connection as research evidence.

Only completed conversations contribute interview evidence.

---

# 10. Sampling Bias Risks

Potential biases include:

1. Recruiting only AI safety enthusiasts.
2. Recruiting only startup founders.
3. Recruiting only people who already know MechGuard.
4. Recruiting only people who agree with the hypothesis.
5. Recruiting only researchers.
6. Recruiting only one geographic region.
7. Recruiting only one organization size.
8. Asking leading questions.

Actively seek contradictory evidence.

---

# 11. Recruitment Success Criteria

The recruitment phase should aim to produce:

1. Multiple independent organizations.
2. Multiple practitioner roles.
3. Multiple transformation workflows.
4. Multiple evaluation maturity levels.
5. At least some participants with production experience.
6. At least some participants who contradict one or more hypotheses.

A participant who contradicts the hypothesis can be more informative than
another participant who simply agrees.

---

# 12. Current Recruitment Status

Target interviews: 10–15

Completed interviews: 0

Relevant practitioner evidence: 0

Problem validation status: Not yet determined.

No validation claim should be made until practitioner evidence has been
collected and analyzed.
