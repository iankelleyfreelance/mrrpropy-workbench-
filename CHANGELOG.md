# Changelog

All notable changes to this project should be documented in this file.

The format is based on Keep a Changelog and the project follows semantic-style
versioning for published package releases.

## [Unreleased]

## [0.1.0]

### Added

- Production readiness checklist in `PRODUCTION.md`.
- Release validation workflow for forced RAW reprocessing checks.
- Packaging and release preparation docs for PyPI publication.

### Changed

- `MRRProData` now delegates plotting, rain-process analysis, and RaProMPro
  processing/load responsibilities into dedicated modules.
- The enforced `mypy` lane now covers the extracted typed subset of the package.

## [0.0.1]

### Added

- Initial packaged release of `mrrpropy`.
