# mrrpropy
MRR PRO code for processing and analysis.

## Repository workflow
This repository keeps the scientific processing code intact while standardizing the developer workflow around it.

- Install the project in editable mode with `uv sync --group dev` or `pip install -e .`.
- Import the package as `mrrpropy`.
- Use the optional CLI entry point as `mrrpro version`.
- Keep scientific algorithm changes confined to the processing modules and treat workflow, packaging, tests, and CI as separate concerns.

## Development
Typical local commands:

```bash
uv sync --group dev
uv run python -c "import mrrpropy"
uv run pytest -m "not slow"
uv run pytest -m slow
uv run mypy
uv run black --check mrrpropy tests
uv run python scripts/benchmark_raprompro.py --quick --repeats 1
```

## Tests
The test suite is organized into:

- fast checks for import and basic data access,
- integration checks for end-to-end workflow behavior,
- slow plotting regressions that write figures under `tests/figures/`.
- generated NetCDF and other non-figure test outputs under `tests/generated/`.

Bundled NetCDF files under `tests/data/` remain the reference fixtures. Generated outputs should go to ignored test output directories, not back into tracked fixture paths.

## Benchmarking
For quick performance checks, use the bundled 10-minute RAW subset:

```bash
uv run python scripts/benchmark_raprompro.py --quick --repeats 1
```

For the full one-hour fixture, pass `--raw-path` explicitly or omit `--quick`.

## Documentation
The repository includes a static documentation site for GitHub Pages.

- Build locally with `python scripts/build_docs.py` after installing `.[docs]`.
- Preview locally with `python -m http.server 8000 --directory site` after the build.
- The landing pages live under `docs/`.
- The API reference is generated with `pdoc` from package docstrings and signatures.
