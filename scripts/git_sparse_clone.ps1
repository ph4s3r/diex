param(
    [string]$RootDir = "C:\dev\sources-to-index\",
    [string]$Project = "defender-docs",
    [string]$Origin = "https://github.com/MicrosoftDocs/$Project.git",
    [string]$Ref = "public",
    [switch]$IncludePDF
)
    # gitlab-example
    # [string]$Project = "docs-gitlab-com",
    # [string]$Origin = "https://gitlab.com/gitlab-org/technical-writing/$Project.git",

    # [string]$Project = "microsoft-365-docs",
    # [string]$Origin = "https://github.com/MicrosoftDocs/$Project.git",
    # [string]$Ref = "public",

# Confirm parameters if not passed as arguments
if (-not $PSBoundParameters.ContainsKey('RootDir')) {
    $confirm = Read-Host "Use default root directory '$RootDir'? (y/n)"
    if ($confirm -eq 'n') {
        $RootDir = Read-Host "Enter root directory"
    }
}

if (-not $PSBoundParameters.ContainsKey('Project')) {
    $confirm = Read-Host "Use default project name '$Project'? (y/n)"
    if ($confirm -eq 'n') {
        $Project = Read-Host "Enter project name"
    }
}

if (-not $PSBoundParameters.ContainsKey('Origin')) {
    $confirm = Read-Host "Use default origin '$Origin'? (y/n)"
    if ($confirm -eq 'n') {
        $Origin = Read-Host "Enter git origin URL"
    }
}

if (-not $PSBoundParameters.ContainsKey('Origin')) {
    $confirm = Read-Host "Use default ref '$Ref'? (y/n)"
    if ($confirm -eq 'n') {
        $Origin = Read-Host "Enter git origin URL"
    }
}

Write-Host "Starting sparse checkout of $Project from $Origin into $RootDir\$Project..."

$originalDir = Get-Location

try {
    Set-Location $RootDir
    git init $Project
    Set-Location $Project
    git remote add origin $Origin
    git config core.sparseCheckout true
    New-Item -ItemType Directory -Force -Path ".git/info" | Out-Null

    # Configure sparse checkout to include markdown and optionally PDF files
    if ($IncludePDF) {
        "**/*.md", "**/*.pdf" | Set-Content -Path ".git/info/sparse-checkout"
    } else {
        "**/*.md" | Set-Content -Path ".git/info/sparse-checkout"
    }

    git pull --depth=1 origin $Ref

    $mdCount = (Get-ChildItem -Filter *.md -Recurse | Measure-Object).Count
    $pdfCount = 0
    if ($IncludePDF) {
        $pdfCount = (Get-ChildItem -Filter *.pdf -Recurse | Measure-Object).Count
    }

    $sizeBytes = (Get-ChildItem -Recurse | Measure-Object Length -Sum).Sum
    $sizeMB = [math]::Round($sizeBytes / 1MB, 2)

    Write-Host ""
    Write-Host "Checkout complete."
    Write-Host ("Markdown files pulled : {0}" -f $mdCount)
    if ($IncludePDF) {
        Write-Host ("PDF files pulled     : {0}" -f $pdfCount)
    }
    Write-Host ("Directory size        : {0} MB ({1:N0} bytes)" -f $sizeMB, $sizeBytes)
}
finally {
    Set-Location $originalDir
    Write-Host "Returned to original directory: $originalDir"
}