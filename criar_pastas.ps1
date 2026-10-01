# Cria as pastas de peças jurídicas em Documentos e registra as instruções para o Claude.
$ErrorActionPreference = 'Stop'
$docs = [Environment]::GetFolderPath('MyDocuments')
$pastas = 'Petição Inicial', 'Contestação', 'Apelação', 'Agravo', 'Réplica', 'Mero Despacho'

foreach ($p in $pastas) {
    $caminho = Join-Path $docs $p
    New-Item -ItemType Directory -Force -Path $caminho | Out-Null
    Write-Host "Pasta pronta: $caminho"
}

# Instruções globais do Claude Code (valem em qualquer sessão no seu computador)
$claudeDir = Join-Path $env:USERPROFILE '.claude'
$claudeMd  = Join-Path $claudeDir 'CLAUDE.md'
$instrucoes = Get-Content -Raw -Encoding UTF8 (Join-Path $PSScriptRoot 'INSTRUCOES_CLAUDE.md')
$marcador = '# Organização dos documentos jurídicos'

New-Item -ItemType Directory -Force -Path $claudeDir | Out-Null
$atual = if (Test-Path $claudeMd) { Get-Content -Raw -Encoding UTF8 $claudeMd } else { '' }
if ($atual -notlike "*$marcador*") {
    Add-Content -Encoding UTF8 -Path $claudeMd -Value ("`r`n" + $instrucoes)
    Write-Host "Instruções adicionadas em: $claudeMd"
} else {
    Write-Host "Instruções já estavam em: $claudeMd"
}

Write-Host ''
Write-Host 'Concluído.'
