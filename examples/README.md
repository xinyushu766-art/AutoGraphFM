# Examples

The main reproducible workflows are exposed through `scripts/`:

```bash
# Real aggregate market benchmark
python scripts/run_stage10b.py --output-dir outputs/stage10b_real --overwrite

# Stability-selected graph validation
python scripts/validate_stage12a.py --overwrite --full

# Complete audited demo pipeline
python scripts/validate_stage4.py --overwrite
```

Generated files are written under `outputs/` and are intentionally excluded from Git.
