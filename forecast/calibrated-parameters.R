# Loading Utah cases
source("calibration/dataprep.R")
utah_cases <- fread("data-raw/covid19-utah-cases.csv")

saved_model <- keras::load_model_hdf5("sir-keras_infections_only")

# Number of observations per chunk
ntimes <- 50
ids <- cbind(
  from = seq(1, nrow(utah_cases)-ntimes, by = 5),
  to   = seq(1 + ntimes, nrow(utah_cases), by = 5)
  )

# Setting up the data
samples <- vector("list", length=nrow(ids))
for (s in seq_along(samples)) {

  s_from <- ids[s, 1]
  s_to   <- ids[s, 2]

  samples[[s]] <- prepare_data_infections_only.default(
    utah_cases$cases[s_from:s_to],
    max_days = 60
    )

}

# Running the simulations
nsims <- 100 # per dataset
nthreads <- 20

calibrated_values <- lapply(
  samples, predict, object = saved_model
  ) |>
  do.call(what=rbind) |>
  as.data.table() |>
  setnames(
    c("init_state", "contact_rate", "transmission_rate", "recovery_rate")
    )

# Assigning the ids
calibrated_values[, id_start := ids[, 1]]
calibrated_values[, id_end := ids[, 2]]
calibrated_values[, initial_cases := utah_cases$cases[id_start]]
calibrated_values[, final_cases := utah_cases$cases[id_end]]

calibrated_values[, contact_rate :=  qlogis(contact_rate)]

# Only finnite values for contact rate
calibrated_subset <- copy(calibrated_values)
calibrated_subset <- calibrated_subset[contact_rate %inrange% c(0, 100)]

# Reasonable recovery and transmission rates
calibrated_subset <- calibrated_subset[
  transmission_rate %inrange% c(0.005, .9) &
    recovery_rate %inrange% c(0.005, .9)
    ]

# Saving the preped-parameters
fwrite(calibrated_subset, "forecast/calibrated-parameters.csv")


# Now, generate a table using knitr of calibrated_subset
# This is the table that will be used in the manuscript
# to show the calibrated parameters


