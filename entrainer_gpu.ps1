# Entraîne le modèle maison sur le GPU (DirectML) et relance avec reprise en cas de plantage.
#   powershell -ExecutionPolicy Bypass -File entrainer_gpu.ps1
param([string]$Taille = "petit", [string]$Sortie = "entrainements/v1", [int]$Epoques = 40)
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
New-Item -ItemType Directory -Force $Sortie | Out-Null
$journal = Join-Path $Sortie "entrainement.log"
$echecs = 0
while ($true) {
    $debut = Get-Date
    & .\.venv-gpu\Scripts\python.exe -m modele.entrainer --donnees donnees --sortie $Sortie `
        --taille $Taille --epoques $Epoques --appareil directml --reprendre *>> $journal
    if ($LASTEXITCODE -eq 0) { Add-Content $journal "=== Entraînement terminé ==="; break }
    # Un plantage après plus de 10 minutes de travail remet le compteur à zéro.
    if (((Get-Date) - $debut).TotalMinutes -gt 10) { $echecs = 0 }
    $echecs++
    Add-Content $journal "=== Plantage (code $LASTEXITCODE), relance n° $echecs avec reprise ==="
    if ($echecs -ge 8) { Add-Content $journal "=== Abandon : 8 plantages rapprochés ==="; break }
    Start-Sleep -Seconds 10
}
