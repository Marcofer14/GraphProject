# Long-running job chains. Each chain runs its steps in order and logs to logs/<chain>.log.
# Launch detached (survives closing the editor / Claude session):
#   powershell -File scripts/start_jobs.ps1 A B C
param([Parameter(Mandatory)][ValidateSet("A", "B", "C", "D", "E")][string]$Chain)

$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".venv\Scripts\python.exe"
$log = "logs\chain_$Chain.log"
New-Item -ItemType Directory -Force logs | Out-Null

function Step([string]$name, [string[]]$cmd, [string]$skipIfExists = "") {
    if ($skipIfExists -and (Test-Path $skipIfExists)) {
        "[$(Get-Date -Format 'HH:mm:ss')] SKIP  $name ($skipIfExists exists)" | Add-Content $log
        return
    }
    "[$(Get-Date -Format 'HH:mm:ss')] START $name" | Add-Content $log
    $argline = ($cmd | ForEach-Object { "`"$_`"" }) -join " "
    cmd /c "`"$py`" $argline >> `"$log`" 2>&1"  # cmd redirection keeps the log UTF-8
    "[$(Get-Date -Format 'HH:mm:ss')] END   $name (exit $LASTEXITCODE)" | Add-Content $log
}

$synth = "-m", "neural_routing.eval.synthetic_benchmark", "--n", "200", "--workers", "3"
switch ($Chain) {
    "A" {  # GPU: TSP-20 -> its benchmark -> CVRP-50 (hours) -> its benchmark
        Step "train tsp20" @("-m", "neural_routing.ml.train", "--config-name", "tsp20")
        Step "bench tsp20" ($synth + @("--problem", "tsp", "--size", "20", "--configs",
            "nearest_neighbor", "savings", "ortools:1", "pyvrp:5", "attention_greedy", "attention")) `
            "results\synthetic_tsp20.md"
        Step "train cvrp50" @("-m", "neural_routing.ml.train", "--config-name", "cvrp50")
        Step "bench cvrp50" ($synth + @("--problem", "cvrp", "--size", "50", "--configs",
            "nearest_neighbor", "savings", "ortools:1", "ortools:10", "pyvrp:10", "attention_greedy",
            "attention", "--checkpoint", "checkpoints/cvrp50_pomo.pt"))
    }
    "B" {  # GPU: CVRP-20 -> its benchmark -> size generalisation 20 -> 50
        Step "train cvrp20" @("-m", "neural_routing.ml.train", "--config-name", "cvrp20")
        Step "bench cvrp20" ($synth + @("--problem", "cvrp", "--size", "20", "--configs",
            "nearest_neighbor", "savings", "ortools:1", "pyvrp:5", "attention_greedy", "attention"))
        Step "bench cvrp50 with cvrp20 model" ($synth + @("--problem", "cvrp", "--size", "50",
            "--configs", "savings", "pyvrp:10", "attention", "--checkpoint",
            "checkpoints/cvrp20_pomo.pt", "--tag", "from_cvrp20"))
    }
    "D" {  # GPU: continue CVRP-50 from epoch 40 to 100, then re-benchmark (same 200 instances)
        Step "train cvrp50 to 100 epochs" @("-m", "neural_routing.ml.train", "--config-name",
            "cvrp50", "train.epochs=100")
        Step "bench cvrp50 e100" ($synth + @("--problem", "cvrp", "--size", "50", "--configs",
            "pyvrp:10", "ortools:10", "attention", "--checkpoint", "checkpoints/cvrp50_pomo.pt",
            "--tag", "e100")) "results\synthetic_cvrp50_e100.md"
    }
    "E" {  # GPU overnight: size curriculum 50 -> 100 -> 150, then the final real-route benchmark
        Step "curriculum CVRP-100" @("-m", "neural_routing.ml.train", "--config-name",
            "cvrp100_amazon")
        Step "curriculum CVRP-150" @("-m", "neural_routing.ml.train", "--config-name",
            "cvrp150_amazon")
        Step "final: 200 real routes" -skipIfExists "results\route_evaluation_n200_final.md" -cmd @(
            "-m", "neural_routing.eval.benchmark", "--mode", "route", "--split", "evaluation",
            "--n", "200", "--workers", "6", "--tag", "final", "--configs", "reference",
            "savings_ls", "ortools:10", "pyvrp:10",
            "attention_ls@checkpoints/cvrp150_pomo_amazon.pt",
            "attention_ls@checkpoints/cvrp50_pomo_amazon.pt",
            "attention_ls@checkpoints/cvrptw50_pomo_amazon.pt")
    }
    "C" {  # CPU: real Amazon routes, then whole station-days
        Step "bench amazon routes" -skipIfExists "results\route_training_n200.md" -cmd @("-m", "neural_routing.eval.benchmark", "--mode", "route",
            "--split", "training", "--n", "200", "--workers", "4", "--configs", "reference",
            "nearest_neighbor", "savings", "ortools:1", "ortools:10", "pyvrp:1", "pyvrp:10")
        Step "bench amazon station-days" @("-m", "neural_routing.eval.benchmark", "--mode",
            "station-day", "--split", "training", "--n", "6", "--max-stops", "1200", "--workers",
            "3", "--configs", "reference", "nearest_neighbor", "savings", "ortools:60", "pyvrp:60")
    }
}
"[$(Get-Date -Format 'HH:mm:ss')] CHAIN $Chain DONE" | Add-Content $log
