# Start job chains as detached processes (created via WMI, so they are not children of the
# terminal or editor and keep running if those close). Usage: powershell -File scripts/start_jobs.ps1 A B C
param([Parameter(ValueFromRemainingArguments)][string[]]$Chains)
if (-not $Chains) { $Chains = @("A", "B", "C") }

$root = Split-Path $PSScriptRoot -Parent
foreach ($c in $Chains) {
    $cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$root\scripts\jobs.ps1`" -Chain $c"
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create `
        -Arguments @{ CommandLine = $cmd; CurrentDirectory = $root }
    "chain $c -> pid $($r.ProcessId) (log: logs\chain_$c.log)"
}
