param(
    [Parameter(Mandatory)][string]$DataDir,
    [Parameter(Mandatory)][string]$StudentGraphRoot,
    [string]$OutputDir = (Join-Path $PSScriptRoot 'results'),
    [switch]$BaselineOnly
)

$env:OMP_NUM_THREADS = '4'
$env:MPLBACKEND = 'Agg'
$features = @(
    'dist_m', 'heading_diff_deg', 'heading_cos', 'heading_valid',
    'along_frac', 'arc_len_m', 'speed_mps', 'dist_rank', 'dist_minus_min_m',
    'n_cand', 'dt_prev_s', 'dt_next_s', 'has_reverse_arc'
)
$arguments = @(
    (Join-Path $PSScriptRoot 'pathsnap_models.py'),
    '--train-candidates', (Join-Path $DataDir 'candidates_train.csv.gz'),
    '--train-reference', (Join-Path $DataDir 'labels_train.csv.gz'),
    '--test-candidates', (Join-Path $DataDir 'candidates_student_test.csv.gz'),
    '--test-reference', (Join-Path $DataDir 'labels_student_test.csv.gz'),
    '--student-graph-root', $StudentGraphRoot,
    '--output', $OutputDir, '--features'
) + $features
if ($BaselineOnly) { $arguments += '--baseline-only' }
python @arguments
exit $LASTEXITCODE
