# A001 — Temporal Analysis

## Status

**Data-limited — temporal EM prediction not demonstrated**

## Question

A001 was designed to test whether training-time internal weight geometry contains predictive information about subsequent emergent misalignment (EM), rather than merely correlating with contemporaneous behavior.

## Available persisted evidence

The A001 archive contains:

- 45 checkpoint-level geometry measurements from steps 10 through 441.
- Geometry extracted from `base_model.model.model.layers.8.mlp.down_proj`.
- Behavioral generations at checkpoints 10, 100, 200, 300, and 441.
- 80 generated responses per behavioral checkpoint.
- Behavioral screening metrics for refusal, medical caution, harmful instruction rate, response length, and related heuristics.

## Critical limitation

The persisted behavioral evaluation is explicitly response-collection-only. The behavioral manifest records no judge metrics and confirms that the Azure judge was not used.

Therefore the archive does **not** contain a validated checkpoint-level EM score at the 45 geometry checkpoints.

Because future-EM labels are unavailable, the following analyses cannot be validly computed from the persisted A001 evidence:

- future EM AUROC
- future EM AUPRC
- predictive temporal classification
- lead-time-to-EM
- early-warning threshold validation
- temporal prediction beyond contemporaneous heuristic correlations

## What remains supported

A001 does support an exploratory description of training-time geometry alongside the five-point behavioral trajectory.

The top singular value increased from approximately 1.34 at step 10 to approximately 11.19 at step 441, while the behavioral screening trajectory changed across the same checkpoints.

These observations are descriptive. They do not establish that geometry predicts EM, causes EM, or provides an operational early-warning signal.

## Scientific conclusion

The A001 pipeline successfully established reproducible training-time geometry extraction and checkpoint-level behavioral response collection.

The stronger temporal hypothesis remains **unresolved by the persisted dataset** because validated EM labels were not collected at the full checkpoint cadence.

A future A001 temporal run would need:

1. behavioral EM scoring at every monitored checkpoint, or a sufficiently dense predefined checkpoint schedule;
2. a fixed judge/evaluation protocol;
3. future-label construction such as EM at `t + delta`;
4. lagged AUROC/AUPRC;
5. lead-time analysis;
6. comparison against contemporaneous EM and training progression;
7. control-model false-positive analysis;
8. preferably multiple random seeds.

## Claim boundary

A001 should therefore be described as:

> **A reproducible exploratory study of training-time weight geometry with sparse behavioral trajectory measurements, not as a demonstrated temporal EM predictor.**
