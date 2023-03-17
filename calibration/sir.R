library(epiworldR)
library(data.table)

N     <- 100
n     <- 1000
ndays <- 100

set.seed(1231)

theta <- data.table(
  preval = rbeta(N, 1, 19),     # Mean 1/(1 + 19) = 0.05
  repnum = rgamma(N, 4, 4/1.5), # Mean 4/(4 / 1.5) = 1.5
  ptran  = rbeta(N, 19, 1),     # Mean 19/(1 + 19) = 0.95
  prec   = rbeta(N, 10, 10*2 - 10) # Mean 10 / (10 * 2 - 10) = .5
)

ans <- vector("list", N)
for (i in 1:N) {
  m <- theta[i,
    ModelSIRCONN(
      "mycon",
      prevalence = preval,
      reproductive_number = repnum,
      prob_transmission = ptran,
      prob_recovery = prec,
      n = n
      )
    ]
  run(m, ndays = ndays)
  ans[[i]] <- get_hist_total(m)
}

