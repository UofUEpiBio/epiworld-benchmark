#!/bin/sh
#SBATCH --job-name=epiworld-benchmark
#SBATCH --output=/uufs/chpc.utah.edu/common/home/vegayon-group1/george/epiworld-benchmark/benchmark.out
#SBATCH --account=vegayon-np
#SBATCH --partition=vegayon-np
/usr/lib64/R/bin/Rscript --vanilla benchmarks.R
