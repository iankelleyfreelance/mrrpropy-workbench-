# Production Readiness

This repository is close to production-ready for supervised batch operation, but
it should be released through a controlled validation gate rather than deployed
directly from day-to-day development changes.

## Recommended rollout

Start with an internal or supervised deployment:

- known input data sources
- monitored logs and output directories
- a rollback path to the previous package version or tagged commit

Do not promote a new version until the release candidate passes the validation
checks below.

## Release checklist

1. Freeze the candidate you intend to deploy.

- Create a release branch or tag from a known-good commit.
- Avoid deploying from a dirty worktree.

2. Recreate the environment.

```bash
uv sync --group dev
uv run python -c "import mrrpropy"
```

3. Run the required validation gates.

Fast lane:

```bash
uv run mypy
uv run pytest -m "not slow" -q
```

Forced end-to-end reprocessing from RAW:

PowerShell:

```powershell
$env:MRRPRO_FORCE_REPROCESS = "1"
uv run pytest tests/mrr/test_1_load_raw.py tests/mrr/test_3_raprompro.py tests/mrr/test_4_plots_raprompro.py tests/mrr/test_5_rain_processes.py tests/mrr/test_6_plot_rain_classification.py tests/mrr/test_8_plot_rain_classification_in_hexagram.py -q
```

Bash:

```bash
MRRPRO_FORCE_REPROCESS=1 uv run pytest tests/mrr/test_1_load_raw.py tests/mrr/test_3_raprompro.py tests/mrr/test_4_plots_raprompro.py tests/mrr/test_5_rain_processes.py tests/mrr/test_6_plot_rain_classification.py tests/mrr/test_8_plot_rain_classification_in_hexagram.py -q
```

4. Review the generated outputs.

- Confirm that the generated NetCDF product is created under `tests/generated/`.
- Review regression metrics and any diagnostic figures when the release touches
  the processing path.
- Compare output naming and destination paths with the production job
  expectations.

5. Confirm operational readiness.

- Input paths and permissions are stable in the target environment.
- Output directories are writable.
- Logs are captured by the scheduler or service manager that will run the job.
- A previous known-good version remains available for rollback.

## CI release gate

Use the dedicated GitHub Actions workflow in
`.github/workflows/release-validation.yml` before promoting a version. That
workflow keeps the default CI fast while still forcing the expensive RAW
reprocessing path for release validation.
