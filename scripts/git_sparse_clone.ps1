# ms-docs project sparse checkout
$rootdir="C:\dev"
$project="powershell-docs"

Set-Location $rootdir
git init $project
Set-Location $project
git remote add origin https://github.com/MicrosoftDocs/$project.git
git config core.sparseCheckout true
New-Item -ItemType Directory -Force -Path ".git/info" | Out-Null
"**/*.md" | Set-Content -Path ".git/info/sparse-checkout"
git pull --depth=1 origin main