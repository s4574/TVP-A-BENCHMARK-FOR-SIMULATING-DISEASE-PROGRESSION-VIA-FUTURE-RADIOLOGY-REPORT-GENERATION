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
