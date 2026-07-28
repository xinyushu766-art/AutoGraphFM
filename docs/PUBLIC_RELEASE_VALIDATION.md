# Public release validation

Validation date: **2026-07-28**

## Executed checks

| Check | Result |
|---|---:|
| Wheel build and isolated target installation | Passed |
| Installed package version | `0.18.0` |
| Automated tests | **108/108 passed** |
| Python `compileall` | Passed |
| Stage 12A full acceptance | **10/10 passed** |
| Stage 12A run ID | `040ddd41a644709a` |
| Candidate-edge Precision@K | **0.8333** |
| Final cross-seed edge Jaccard | **1.0000** |
| Internal Markdown links | Passed |
| TOML and GitHub workflow YAML parsing | Passed |
| Forbidden account references and secret-pattern scan | Passed |
| Cache/build/database artifacts in final archive | Required to be zero |

## Static-lint limitation

Ruff was not available from the execution environment's package index, so this release does not claim a locally executed Ruff pass. The repository includes Ruff configuration and the CI environment can run it when the package is available. Syntax compilation, automated tests, whitespace checks, and import/build validation were executed locally.

## Evidence boundary

Passing this validation authorizes the repository as a reproducible engineering and portfolio artifact. It does not authorize a real vehicle-level graph superiority claim; that remains `NO-CLAIM` until the complete real vehicle panel passes the locked temporal and graph-control protocol.
