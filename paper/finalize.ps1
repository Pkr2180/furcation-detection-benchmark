# Wait for the saveetha ablation calls, download results, re-run every analysis, rebuild figures/tables/manuscript, audit.
#   powershell -File paper/finalize.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONUTF8 = '1'; $env:MODAL_PROFILE = 'saveetha'
while ($true) {
    $st = python spawn_dino.py --status logs/spawned_saveetha.json
    $run = ($st | Select-String 'running').Count; $bad = ($st | Select-String 'FAILED').Count
    "$(Get-Date -Format HH:mm) running=$run failed=$bad"
    if ($bad -gt 0) { $st | Select-String 'FAILED'; exit 2 }
    if ($run -eq 0) { break }
    Start-Sleep -Seconds 300
}
foreach ($v in 'no_mirror', 'no_coords', 'no_context') { modal volume get --force furcation-v4 "/results/$v" res_all/ }
python analyze.py res_all manifest.json res_final.json
python sensitivity.py res_all manifest.json folds.json sens_final.json
python paper/stats_extra.py res_all manifest.json paper/stats_extra.json
python paper/make_figs_tables.py
python paper/build_manuscript.py
python paper/audit_manuscript.py
"FINALIZE DONE $(Get-Date -Format HH:mm)"
