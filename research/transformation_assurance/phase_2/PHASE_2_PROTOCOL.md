# MechGuard Phase 2: Predictive Transformation Assurance Protocol

## 1. Research Objective

Phase 2 tests whether internal model-transformation signals can predict meaningful behavioral regressions before the corresponding behavioral evaluation suite is executed.

The target is not general model safety, alignment, intent, deception, or certification.

The specific research question is:

> Can a predictor based on internal transformation signals identify whether a specified behavioral evaluation is likely to exhibit a meaningful regression after a model transformation, when evaluated on transformation conditions and model families not used to develop the predictor?

The intended downstream use, if supported, is conservative evaluation triage: identifying transformations that warrant deeper evaluation while abstaining when evidence is outside the validated domain.

## 2. Primary Hypothesis

H1: Internal transformation signals contain predictive information about subsequent behavioral regression beyond simple transformation-size and training-state baselines.

## 3. Generalization Hypothesis

H2: A predictor frozen using one transformation regime retains useful predictive performance on at least one transformation regime not used during predictor development.

## 4. Cross-Model Hypothesis

H3: A predictor frozen on one base-model family retains useful predictive information on a second base-model family without feature or threshold retuning.

## 5. Baseline Hypothesis

H4: Internal transformation features provide predictive information beyond simple baselines including parameter/update magnitude, divergence from the base model, training/validation loss, transformation metadata, and a small predefined behavioral screen.

## 6. Conservative Decision Hypothesis

H5: A selective predictor with an abstention region can identify a subset of transformations with controlled observed regression risk while sending uncertain or out-of-distribution transformations to full evaluation.

No claim of universal safety inheritance or unconditional evaluation skipping will be tested.

## 7. Primary Outcome

The primary behavioral outcome is a predefined meaningful regression between a frozen base-model reference and the transformed model.

A behavioral regression must be defined before predictor development and evaluated using a frozen behavioral dataset.

For safety-specific analysis, a safety regression is a predefined harmful-prompt refusal failure in which the base reference satisfies the frozen refusal criterion and the transformed model fails it.

Behavioral change in either direction is not automatically considered a regression.

## 8. Experimental Separation

Phase 2 will maintain strict separation between:

1. Predictor-development data
2. Predictor-validation data
3. Final held-out evaluation data

Final held-out behavioral labels must not be used to choose features, thresholds, transformations, prompts, or model-specific decision rules.

The final test set must remain frozen until the predictor and decision policy are locked.

## 9. Predictor Development

The initial predictor-development regime will use a controlled transformation family.

The exact transformation family, base model, datasets, seeds, and hyperparameter ranges will be recorded in a frozen experiment manifest before execution.

Features must be measurable before the behavioral outcome is evaluated.

Candidate feature families may include:

1. Relative parameter/update norm
2. Parameter-space divergence
3. Weight/update cosine similarity
4. Singular-value and spectral changes
5. Effective-rank or related geometry measures
6. Other explicitly preregistered transformation-level internal statistics

Features may not be added after inspecting final held-out behavioral labels.

## 10. Required Baselines

The following baselines will be evaluated where technically applicable:

1. Update/weight norm
2. KL divergence or equivalent output divergence on a frozen unlabeled calibration corpus
3. Training or validation loss
4. Transformation metadata
5. A small frozen behavioral screen
6. Random/permutation baseline

The purpose is to determine whether internal geometry contributes information beyond simpler alternatives.

## 11. Transformation Generalization

At least one transformation family used for final evaluation must be excluded from predictor development.

Candidate families include:

1. LoRA/SFT
2. Preference optimization
3. Quantization
4. Model merging
5. Distillation

The final transformation-family split will be frozen before final evaluation.

## 12. Model Generalization

At least one base-model family must be excluded from predictor development and used only for held-out evaluation.

The held-out model must not receive feature-specific or threshold-specific retuning.

## 13. Positive and Negative Examples

The experimental dataset must contain both:

1. Transformations producing predefined behavioral regression
2. Transformations not producing the predefined regression

The method used to generate or obtain positive examples must be documented.

Positive-example construction must not leak the final behavioral label into predictor features.

## 14. Behavioral Evaluation

Behavioral prompts and scoring criteria must be frozen before final predictor evaluation.

The evaluation should include:

1. Safety/refusal behavior
2. Relevant benign behavior
3. Capability or task behavior where applicable
4. Item-level outcomes rather than aggregate scores alone

Raw model outputs must be retained for auditability.

Any heuristic classifier must be validated against manually inspected outputs before final interpretation.

## 15. Metrics

Primary predictive metrics:

1. AUROC
2. AUPRC
3. Brier score or another appropriate calibration metric
4. Recall at specified operational coverage
5. False-negative rate
6. Risk-coverage curve
7. Abstention/coverage rate

Metrics must be reported with uncertainty estimates where sample size permits.

Performance must also be reported separately by transformation family and model family.

## 16. Rare-Event Handling

Safety regressions may be rare.

Therefore:

1. Accuracy must not be used as the primary predictive metric.
2. AUPRC must be reported when class imbalance exists.
3. False-negative behavior must be explicitly reported.
4. Confidence intervals or appropriate uncertainty estimates must be reported where feasible.
5. A predictor that produces no positive predictions must not be interpreted as successful simply because it has zero false positives.

## 17. Selective Decision Policy

The intended policy is not binary safe/unsafe classification.

The candidate policy is:

LOW-RISK / WITHIN-CALIBRATION-DOMAIN:
Potentially eligible for reduced evaluation.

UNCERTAIN:
Require additional evaluation.

OUT-OF-DISTRIBUTION:
Abstain and require full evaluation.

The operational thresholds must be selected using development/calibration data only and frozen before final testing.

## 18. Leakage Controls

The experiment must guard against:

1. Transformation-family leakage
2. Model-family leakage
3. Seed leakage
4. Dataset leakage
5. Prompt leakage
6. Hyperparameter leakage
7. Post-hoc threshold selection
8. Feature selection using final labels
9. Outcome-derived transformation metadata
10. Accidental reuse of final evaluation examples during predictor development

## 19. Required Comparisons

The final report must compare:

1. MechGuard internal-signal predictor
2. Each simple baseline
3. Combined models where preregistered
4. Random/permutation baseline

The report must state whether internal signals add measurable information beyond the strongest simple baseline.

## 20. Failure Criteria

Phase 2 will be considered unsuccessful as a general predictive claim if one or more of the following occurs:

1. Performance collapses on unseen transformation families.
2. Performance collapses on the held-out model family.
3. Internal features do not outperform simple baselines.
4. Predictive performance depends on post-hoc feature or threshold tuning.
5. The predictor exhibits unacceptable false-negative behavior under the tested operating point.
6. The predictor cannot distinguish regression from non-regression beyond chance.
7. Results are explained adequately by transformation metadata alone.
8. Apparent performance disappears under leakage or confound controls.

A negative result is scientifically valid and will be reported without changing the protocol after observing outcomes.

## 21. Success Criteria

A strong positive result requires:

1. Predictive performance above chance.
2. Generalization to at least one unseen transformation family.
3. Generalization to a held-out model family.
4. Improvement over relevant simple baselines.
5. Explicitly measured calibration or selective-risk behavior.
6. Conservative false-negative analysis.
7. No identified major leakage explaining the result.

Meeting these criteria does not establish universal safety prediction or certification.

## 22. Reproducibility

Every final transformation must record:

1. Base model
2. Model revision
3. Transformation family
4. Dataset identity and hash
5. Dataset split
6. Random seed
7. Hyperparameters
8. Software environment
9. Hardware environment
10. Transformation artifact identifier
11. Feature configuration
12. Behavioral evaluation version

All final artifacts must be committed to the research branch.

## 23. Phase 2 Interpretation Standard

Possible conclusions are:

A. Evidence supports generalizable predictive information.

B. Evidence supports predictive information only within a restricted transformation/model regime.

C. Internal signals are primarily descriptive and do not provide useful predictive information beyond simpler baselines.

D. Results are inconclusive because of insufficient data, rare positive outcomes, or experimental limitations.

The experiment must not force a positive conclusion.

## 24. Product Relevance

Only if technical evidence supports generalization and conservative risk control should the results be interpreted as supporting a product hypothesis.

The potential product use is evaluation triage and transformation-impact assessment.

The experiment does not test or establish:

1. Universal AI safety
2. Detection of hidden goals
3. Detection of deception
4. Detection of intent
5. Formal safety guarantees
6. Regulatory certification
7. Safe deployment without behavioral evaluation

## 25. Protocol Freeze

This document defines the Phase 2 experimental standard.

After the final held-out evaluation begins, changes to the primary hypothesis, final labels, held-out prompts, primary features, or decision thresholds require explicit versioning and must not overwrite the original protocol.

