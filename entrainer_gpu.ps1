# Entraîne le modèle maison sur le GPU (DirectML) et relance avec reprise en cas de plantage.
#   powershell -ExecutionPolicy Bypass -File entrainer_gpu.ps1
# Un entraînement DirectML occupe environ 11,6 Go de mémoire vive : n'en lancer qu'un à la fois.
param([string]$Taille = "petit", [string]$Sortie = "entrainements/v2", [int]$Epoques = 40)
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
New-Item -ItemType Directory -Force $Sortie | Out-Null
$journal = Join-Path $Sortie "entrainement.log"
$options = "--donnees donnees --sortie $Sortie --taille $Taille --epoques $Epoques --appareil directml --reprendre " +
           "--lr 6e-4 --montee 3000 --amorce-enonces 400 --amorce-pas 3500 --debut-augmentation 8000 --eval-tous-les 1000"
$echecs = 0
while ($true) {
    $debut = Get-Date
    # Redirection faite par cmd : le journal reste en UTF-8 (PowerShell l'écrirait en UTF-16).
    cmd /c ".venv-gpu\Scripts\python.exe -m modele.entrainer $options >> `"$journal`" 2>&1"
    if ($LASTEXITCODE -eq 0) { Add-Content $journal "=== Entrainement termine ===" -Encoding utf8; break }
    # Un plantage après plus de 10 minutes de travail remet le compteur à zéro.
    if (((Get-Date) - $debut).TotalMinutes -gt 10) { $echecs = 0 }
    $echecs++
    Add-Content $journal "=== Plantage (code $LASTEXITCODE), relance $echecs avec reprise ===" -Encoding utf8
    if ($echecs -ge 8) { Add-Content $journal "=== Abandon : 8 plantages rapproches ===" -Encoding utf8; break }
    Start-Sleep -Seconds 10
}
