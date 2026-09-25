# Query-Conditioned Clinical World Modeling (anonymous release)

Code, environment, schema, and decision traces for the ICLR submission.
Data: derived from MIMIC-IV / MIMIC-IV-Note (PhysioNet credentialed access required).
Raw report/note text and model-generated report caches are NOT included per the data use agreement;
build scripts reconstruct all artifacts from credentialed MIMIC access. Set `VP_ROOT` to the repo root
and provide API credentials via `llm_api/models.yaml` (see `.example`).

- `analysis/` featurizers, builders, scoring, renderers, figures
- `playground/` typed auto-research environment + LLM-policy driver
- `traces/` all auto-research decision traces (actions, scores, policy reasoning)
- `analysis/eval_schema_auto_train.json` frozen evaluation schema

## Data access: two-tier release (MIMIC-IV DUA)

This public repository contains **code, prompts, the findings schema, decision traces, and aggregate results only**. All patient-level derivatives of MIMIC-IV -- per-case prediction records, per-case probabilities, sealed-set identifiers (hadm_id/charttime keys), the extractor-sensitivity per-case file, the same-study leakage audit, and annotation files keyed by case identifiers -- are **not** distributed here: under the PhysioNet Data Use Agreement they will be released as a *derived-data project on PhysioNet*, accessible to credentialed MIMIC-IV users under the same DUA. Aggregate statistics (paired bootstrap results, per-namespace tables, calibration/faithfulness summaries, all decision traces) contain no patient-level data and are included directly.
