# Releasing

This document describes how to prepare and publish the `mrrpropy` package to PyPI.

## One-time setup

1. Create the project on PyPI and, optionally, on TestPyPI.
2. Configure Trusted Publishing for this repository on each index.

For GitHub Actions Trusted Publishing, PyPI requires:

- the GitHub repository owner
- the GitHub repository name
- the exact workflow filename authorized to publish

For this repository, the publish workflow file is:

- `.github/workflows/publish-package.yml`

Recommended GitHub environments:

- `testpypi`
- `pypi`

## Release flow

1. Update `mrrpropy/__init__.py` with the release version.
2. Update `CHANGELOG.md`.
3. Run the release validation checklist from `PRODUCTION.md`.
4. Commit the release candidate.
5. Create an annotated tag matching the package version, for example `v0.0.1`.
6. Push the tag.
7. Create a GitHub Release from that tag and publish it.
8. The publish workflow will build the package, validate the distributions, and
   upload them to PyPI using Trusted Publishing.

## TestPyPI dry run

Use the manual GitHub Actions workflow dispatch for
`.github/workflows/publish-package.yml` to publish the current ref to TestPyPI.

This is the safest way to validate packaging metadata, README rendering, and
index upload behavior before a real PyPI release.

## Local packaging commands

```bash
uv build
uvx twine check dist/*
```

## Notes

- Prefer publishing only from tagged commits and GitHub Releases.
- Keep `release-validation.yml` green before publishing.
- PyPI Trusted Publishing with `pypa/gh-action-pypi-publish` automatically
  produces release attestations.
