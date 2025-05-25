# general project sparse checkout
$rootdir="C:\dev"
$project="docs-gitlab-com"
$origin="https://gitlab.com/gitlab-org/technical-writing/$project.git"

Write-Host "Starting sparse checkout of $project from $origin into $targetDir..."

Set-Location $rootdir
git init $project
Set-Location $project
git remote add origin $origin
git config core.sparseCheckout true
New-Item -ItemType Directory -Force -Path ".git/info" | Out-Null
"**/*.md" | Set-Content -Path ".git/info/sparse-checkout"
git pull --depth=1 origin main

$mdCount = (Get-ChildItem -Filter *.md -Recurse | Measure-Object).Count          # count

$sizeBytes = (Get-ChildItem -Recurse | Measure-Object Length -Sum).Sum           # bytes
$sizeMB    = [math]::Round($sizeBytes / 1MB, 2)                                  # MB (2 dp)

Write-Host ""
Write-Host "Checkout complete."
Write-Host ("Markdown files pulled : {0}" -f $mdCount)
Write-Host ("Directory size        : {0} MB ({1:N0} bytes)" -f $sizeMB, $sizeBytes)