source("calibration/dataprep.R")

calibrated_parameters <- fread("forecast/calibrated-parameters.csv")
utah_cases <- fread("data-raw/covid19-utah-cases.csv")

N     <- 2e4
n     <- 2000
ndays <- 50
ncores <- 20

set.seed(1231)

theta <- calibrated_parameters

data.table(
  preval = rbeta(N, 1, 19),        # Mean 10/(10 + 190) = 0.05
  crate  = rgamma(N, 4, 4/1.5),    # Mean 4/(4 + 1.5) = 1.5
  ptran  = rbeta(N, 7, 3),        # Mean 7/(3 + 7) = 0.7
  prec   = rbeta(N, 10, 10*2 - 10) # Mean 10 / (10 * 2 - 10) = .5
)
theta[, hist(crate)]

matrices <- parallel::mclapply(1:N, FUN = function(i) {

  m <- theta[i,
      ModelSIRCONN(
        "mycon",
        prevalence        = preval,
        contact_rate      = crate,
        transmission_rate = ptran,
        recovery_rate     = prec, 
        n                 = n
        )
      ]

  # Avoids printing
  verbose_off(m)

  run(m, ndays = ndays)

  # Using prepare_data
  prepare_data(m)

}, mc.cores = ncores)


# Keeping only the non-null elements
is_not_null <- intersect(
  which(!sapply(matrices, inherits, what = "error")),
  which(!sapply(matrices, \(x) any(is.na(x))))
  )
matrices <- matrices[is_not_null]
theta    <- theta[is_not_null,]

N <- length(is_not_null)
