# Workbench

Local working area for using `mrrpropy` without mixing user analyses with the
package source tree or the test fixtures.

Suggested layout:

- `data/`: local raw or processed datasets
- `scripts/`: ad hoc processing or plotting scripts
- `notebooks/`: exploratory notebooks
- `output/`: generated figures, NetCDF files and CSV outputs
- `config/`: local configuration files

This directory is ignored by git.

Example:

```powershell
.\.venv311\Scripts\python workbench\scripts\plot_daily_quicklooks.py
```

This generates PNG quicklooks under `workbench/output/quicklooks/...` for all
hourly NetCDF files found in the configured input directory.

To batch-process hourly files into RaProMPro NetCDF products:

```powershell
.\.venv311\Scripts\python workbench\scripts\process_daily_raprompro.py
```

To process a single hourly file:

```powershell
.\.venv311\Scripts\python workbench\scripts\process_one_raprompro.py --input-file workbench\data\mrrpro81\2025\10\29\20251029_120000.nc
```

To generate quicklooks from one processed `*_raprompro.nc` product:

```powershell
.\.venv311\Scripts\python workbench\scripts\plot_one_raprompro_quicklooks.py --input-file workbench\output\raprompro\2025\10\29\20251029_120000_raprompro.nc
```

To generate sampled profile figures from one processed `*_raprompro.nc` product:

```powershell
.\.venv311\Scripts\python workbench\scripts\plot_sampled_profiles.py --input-file workbench\output\raprompro\2025\10\29\20251029_190000_raprompro.nc --step-minutes 10
```

To generate the rain-classification figures used in `test_6` for two different
layers:

```powershell
.\.venv311\Scripts\python workbench\scripts\plot_rain_classification_regions.py --input-file workbench\output\raprompro\2025\10\29\20251029_190000_raprompro.nc
```

To launch it in the background from PowerShell and keep using the terminal:

```powershell
Start-Process -FilePath ".\.venv311\Scripts\python.exe" -ArgumentList "workbench\scripts\process_daily_raprompro.py" -RedirectStandardOutput "workbench\output\raprompro_run.log" -RedirectStandardError "workbench\output\raprompro_run.err.log"
```
