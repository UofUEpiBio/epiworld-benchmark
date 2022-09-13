#!/bin/sh
#SBATCH --job-name=epiworld-benchmark
#SBATCH --output=/uufs/chpc.utah.edu/common/home/vegayon-group1/george/epiworld-benchmark/benchmark.out
#SBATCH --account=vegayon-np
#SBATCH --partition=vegayon-np

library(slurmR)
library(data.table)

project_path <- "/uufs/chpc.utah.edu/common/home/vegayon-group1/george/epiworld-benchmark/"

bfiles <- list.files(pattern = "*.o$", full.names = FALSE)

nrep  <- 1000
sizes <- c(1e4, 1e5, 1e6, 1e7)

res <- parallel::mclapply(
  X        = 1:nrep,
  mc.cores = 40,
  FUN      = \(i) {

    res <- NULL
    for (s in sizes) {

      # Measuring time in seconds
      t0 <- proc.time()
      system(sprintf("./%s %i %i", bfiles[1], 1, s))
      t1 <- proc.time()

      # Appending time
      res <- rbind(
        res,
        data.table(
          i    = i,
          size = s,
          time = (t1 - t0)[3]
        )
      )

    }

    res

  })

# compiling into single data.frame
res <-rbindlist(res)

fwrite(res, paste0(project_path, "benchmark.csv"))



