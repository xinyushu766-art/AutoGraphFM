# Stage 4 Validation Report

- Validation status: **PASSED**
- Stage 4 run ID: `bca30066d366ef05`
- Deployment gate: **BLOCK**
- Graph claim: **NO-CLAIM**
- Evidence scope: **DEMO_ONLY**

## Acceptance checks

- PASS — `required_artifacts_present`
- PASS — `malformed_row_quarantined`
- PASS — `duplicate_key_audited`
- PASS — `panel_has_multiple_series`
- PASS — `panel_observed_ratio`
- PASS — `graph_has_no_isolates`
- PASS — `graph_is_mostly_connected`
- PASS — `validation_selected_models_exist`
- PASS — `production_forecast_built`
- PASS — `monitoring_status_is_explicit`
- PASS — `forecast_sanity_has_no_alerts`
- PASS — `demo_evidence_is_not_release_ready`
- PASS — `graph_claim_gate_is_conservative`
- PASS — `standalone_html_is_complete`

## Gate outcomes

- Blocking reasons: `['global_sales_distribution_drift']`
- Warning reasons: `['series_level_shift_or_missingness']`
- Validation-selected models: `{'1': 'last_value', '3': 'last_value', '6': 'moving_average_6'}`

The validation checks that adverse demo outcomes are surfaced rather than suppressed. Demo rankings are not evidence about the user's real automotive data.
