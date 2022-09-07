#!/bin/sh
#SBATCH --job-name=epiworld-benchmark
#SBATCH --output=/uufs/chpc.utah.edu/common/home/vegayon-group1/george/epiworld-benchmark/benchmark.out
#SBATCH --account=vegayon-np
#SBATCH --partition=vegayon-np

library(slurmR)

bfiles <- list.files(pattern = "*.o$", full.names = FALSE)

for (s in c(1e3, 1e4, 1e5, 1e6, 1e7))
  system(sprintf("./%s %i %i", bfiles[1], 100, s))



