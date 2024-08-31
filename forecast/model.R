source("calibration/dataprep.R")

calibrated_parameters <- fread("forecast/calibrated-parameters.csv")
utah_cases <- fread("data-raw/covid19-utah-cases.csv")

n     <- 200000
ndays <- 50
ncores <- 20

theta <- calibrated_parameters

setnames(
  theta,
  1:4,
  c("preval", "crate", "ptran", "prec"),
  )


# Replicate each row of theta 200 times-> ~10,000 rows
nreplicates <- 10000 %/% nrow(theta)
theta <- theta[rep(1:nrow(theta), each = nreplicates),]

set.seed(1231)

matrices <- parallel::mclapply(1:N, FUN = function(i) {

  # Figuring out the prevalence
  theta[i, preval] 

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
