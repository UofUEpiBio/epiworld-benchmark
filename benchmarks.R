#!/bin/sh
#SBATCH --job-name=epiworld-benchmark
#SBATCH --output=/uufs/chpc.utah.edu/common/home/vegayon-group1/george/epiworld-benchmark/benchmark.out
#SBATCH --account=vegayon-np
#SBATCH --partition=vegayon-np

library(slurmR)

bfiles <- list.files(pattern = "*.o$", full.names = FALSE)
system(sprintf("./%s %i", bfiles[1], 100))


