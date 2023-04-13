#!/bin/sh
#SBATCH --mem=32gb
#SBATCH --account=vegayon-np
#SBATCH --partition=vegayon-shared-np
#SBATCH --time=24:00:00
#SBATCH --mail-user=george.vegayon@utah.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logfile.log

module load R/4.2.2
R --debugger=valgrind --debugger-args="--track-origins=yes" -f sir.R 
#R -f sir.R 



