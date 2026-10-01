# MechGuard — Practitioner Problem Validation Interview Guide

## Purpose

This interview guide is designed to understand how practitioners actually
handle model transformation, evaluation, safety validation, and evidence
reuse.

The interview is discovery-first.

Do not introduce MechGuard at the beginning of the conversation.

Do not attempt to convince the participant that the problem exists.

Do not ask leading questions that imply the expected answer.

---

# 1. Opening

## Introduction

"Thanks for taking the time. I'm researching how teams handle model
adaptation and evaluation in practice.

I'm specifically interested in what happens when an existing model is
fine-tuned, adapted, quantized, merged, distilled, or otherwise modified.

I'm not trying to sell anything here. I mainly want to understand how your
current workflow works and where, if anywhere, it becomes difficult."

---

# 2. Participant Context

### Q1. Role

"What is your current role, and how does it interact with model development,
evaluation, safety, or deployment?"

Record:

1. Role
2. Team
3. Responsibilities
4. Years / level of relevant experience

---

### Q2. Model involvement

"What kinds of models does your team work with?"

Follow up only if relevant:

1. Foundation models
2. Open-weight models
3. Proprietary models
4. LLMs
5. Multimodal models
6. Traditional ML models

---

### Q3. Transformation involvement

"Do you or your team modify existing models before deploying them?"

If yes:

"What was the most recent example?"

Do not immediately list transformation types.

Let the participant describe the workflow first.

If necessary, clarify whether the transformation involved:

1. Fine-tuning
2. LoRA / PEFT
3. Quantization
4. Model merging
5. Distillation
6. Model editing
7. Other adaptation

---

# 3. Reconstruct a Recent Workflow

## Q4. Last transformation

"Can you walk me through the most recent time you took an existing model
and modified it?"

Capture:

1. Starting model
2. Reason for modification
3. Transformation
4. Dataset or adaptation method
5. Evaluation
6. Deployment decision

---

## Q5. Before transformation

"Before you modified the model, what evidence or evaluations did you
already have?"

Probe:

1. Capability benchmarks
2. Safety evaluations
3. Security testing
4. Red-team results
5. Human evaluation
6. Production monitoring
7. Model card / documentation
8. Internal test suites

Do not suggest categories unless the participant needs clarification.

---

## Q6. After transformation

"What did you evaluate after the transformation?"

Then ask:

"How did you decide what to evaluate?"

Capture the actual decision process.

---

# 4. Evidence Reuse

## Q7. Reusing previous evidence

"Did any of the evaluation results from the original model get reused
after the transformation?"

If yes:

"How did you decide that those results were still relevant?"

Capture:

1. Formal rule
2. Internal policy
3. Transformation type
4. Model similarity
5. Human judgment
6. Risk level
7. Previous experience
8. Other reasoning

---

## Q8. Skipped evaluations

"Were there any evaluations you considered but decided not to rerun?"

If yes:

"Why were they skipped?"

This is a critical question.

Capture:

1. Cost
2. Time
3. Low perceived risk
4. Evaluation redundancy
5. Existing evidence
6. Resource constraints
7. Deployment urgency
8. Other reason

---

## Q9. Complete rerun

"Are there situations where you rerun the entire evaluation suite?"

If yes:

"What causes you to choose a complete rerun?"

Capture the trigger.

---

# 5. Operational Cost

## Q10. Time

"Roughly how long does the evaluation process take after a model
transformation?"

If the participant can estimate:

1. Hours
2. Days
3. Weeks

Record uncertainty explicitly.

---

## Q11. Compute

"Does this evaluation process consume significant compute?"

If yes:

"How do you estimate or track that cost?"

Do not pressure the participant for confidential numbers.

---

## Q12. Engineering effort

"How many people are typically involved in deciding whether the transformed
model is ready?"

Capture:

1. Roles
2. Approximate people
3. Review process
4. Approval process

---

# 6. Deployment Decisions

## Q13. Deployment readiness

"Who ultimately decides that a transformed model has enough evidence to
deploy?"

Follow up:

"What information do they need to make that decision?"

---

## Q14. Confidence

"Have you ever had a situation where you were unsure whether previous
evaluation evidence was still sufficient after modifying a model?"

If yes:

"What did you do?"

This is more valuable than asking whether they think such uncertainty
could happen.

---

# 7. Incidents and Surprises

## Q15. Behavioral change

"Have you ever discovered that a transformed model behaved differently
from the original in a way you did not expect?"

If yes:

"How did you discover it?"

Then:

"What happened afterward?"

Capture:

1. Transformation
2. Expected behavior
3. Observed behavior
4. Detection method
5. Evaluation gap
6. Impact
7. Remediation

---

## Q16. Safety or security change

"Have you ever seen a safety, security, or policy-related behavior change
after modifying a model?"

If yes:

"Was that discovered before or after deployment?"

Do not assume that the change was caused by the transformation.

Record the participant's explanation separately from the observed facts.

---

# 8. Current Workarounds

## Q17. Existing process

"When you need confidence that a modified model is safe enough to deploy,
what do you currently do?"

Let the participant describe the process.

Then ask:

"What parts of that process are automated?"

"What parts require manual work?"

---

## Q18. Tools

"What tools do you use for this?"

Capture:

1. Evaluation frameworks
2. Model registries
3. Experiment tracking
4. Safety tooling
5. Red-team tooling
6. Internal systems
7. Scripts
8. Other infrastructure

---

## Q19. Biggest limitation

"If you could remove one painful part of this workflow, what would it be?"

This should be open-ended.

Do not mention MechGuard.

---

# 9. Prioritization

## Q20. Frequency

"How often does this kind of model transformation and revalidation happen?"

Capture:

1. Daily
2. Weekly
3. Monthly
4. Occasionally
5. Rarely

Prefer the participant's own description if it does not fit these categories.

---

## Q21. Importance

"How important is this problem compared with the other engineering or
deployment problems your team deals with?"

Follow up:

"Why?"

---

## Q22. Consequences

"What happens if the evaluation process is insufficient?"

Capture concrete consequences:

1. Deployment delay
2. Rollback
3. Incident
4. Security issue
5. Compliance issue
6. Engineering rework
7. Customer impact
8. No meaningful consequence

---

# 10. Existing Budget and Ownership

## Q23. Existing spending

"Does your team already spend engineering time, compute, or budget on
model evaluation and validation?"

If yes:

"What is that spending primarily used for?"

---

## Q24. Ownership

"Which team owns this workflow?"

Do not assume that AI safety, security, ML platform, or governance owns it.

---

## Q25. Buying / building

"When this workflow is painful, how does your organization normally address
that kind of problem?"

Possible outcomes:

1. Build internally
2. Buy a tool
3. Extend existing infrastructure
4. Ignore it
5. Change the process
6. Other

Only record what the participant actually reports.

---

# 11. Final Open Questions

## Q26.

"What did I not ask about model transformation or evaluation that you think
is important?"

---

## Q27.

"If you had to explain the hardest part of validating a modified model to
another engineer, what would you tell them?"

---

# 12. Optional Concept Test

Only use this section AFTER the workflow and pain discovery questions.

Do not use it if the participant has not described a relevant workflow.

Briefly describe the concept without selling it:

"We are exploring whether there could be a system that keeps track of the
relationship between a model transformation and the evidence available for
the resulting model, potentially helping teams determine what needs
revalidation."

Then ask:

### Q28.

"How does that compare with how your team handles this today?"

### Q29.

"What would such a system need to do to be useful in your workflow?"

### Q30.

"What would make it unnecessary?"

### Q31.

"What would prevent your team from adopting something like this?"

Do not treat positive reactions as problem validation.

---

# 13. Interviewer Rules

## Never lead

Avoid:

"Do you struggle with expensive revalidation?"

Prefer:

"How do you currently handle revalidation?"

---

## Ask for examples

Prefer:

"Tell me about the last time..."

over:

"Would you ever..."

---

## Ask for numbers

Prefer:

"How long did the last evaluation take?"

over:

"Was it expensive?"

---

## Separate facts from interpretation

Record:

1. What happened.
2. What the participant believes happened.
3. What the interviewer infers.

Do not combine them.

---

## Capture negative evidence

If a participant says:

"We rerun everything automatically and it takes almost no time."

Record it.

Contradictory evidence is important.

---

# 14. Interview Output

After every interview, produce:

1. Participant profile
2. Relevant workflow
3. Transformation event
4. Evaluation process
5. Evidence reuse process
6. Pain points
7. Operational cost
8. Existing workaround
9. Incidents
10. Decision owner
11. Evidence classification
12. Strongest quote
13. Strongest contradiction
14. Follow-up question

---

# 15. Evidence Discipline

Do not count these as equivalent:

"That sounds useful."

and:

"We currently spend three days rerunning our evaluation suite after every
fine-tune."

The first indicates hypothetical interest.

The second provides evidence about an existing workflow and operational cost.

The strongest validation comes from repeated concrete evidence across
independent practitioners.

---

# 16. Interview Goal

The goal of each interview is not to obtain agreement.

The goal is to discover:

Actor
→ Transformation
→ Evaluation workflow
→ Evidence reuse decision
→ Pain
→ Cost
→ Consequence
→ Existing workaround
→ Unmet need

If any link is absent, record the absence rather than filling it with
assumptions.
