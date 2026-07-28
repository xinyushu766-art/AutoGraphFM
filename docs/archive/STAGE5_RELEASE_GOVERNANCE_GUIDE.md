# Stage 5 — Release Governance

## Purpose

Stage 5 prevents an executable forecast from becoming an operational deployment
artifact without an explicit policy decision. It reads the Stage 4 manifest and
separately evaluates engineering status, monitoring status, evidence scope, and
an optional graph-claim requirement.

## Production mode

Production mode emits `deployment_bundle/` only when every configured condition
passes. By default it requires:

- engineering status `pass`;
- deployment status `pass` (or `warn` only when explicitly accepted);
- evidence scope `user_data`;
- no graph claim unless the policy explicitly requires one.

A blocked release writes the decision and reasons but does not create a deployable
bundle. There is no force flag that silently changes a blocked decision into an
approved one.

## Shadow mode

Shadow mode is intended for integration tests and challenger evaluation. It may
create `shadow_bundle/`, but its manifest always records `deployable=false`.
This protects demonstration and incomplete evidence from accidental promotion.

## Bundle contents

The governed packet contains the production forecast, calibration, model card,
held-out selected-model metrics, monitoring report, HTML audit, Stage 4 manifest,
and `SHA256SUMS.txt`. The checksum file is independently verifiable.
