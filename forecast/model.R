# Loading Utah cases
source("calibration/dataprep.R")
utah_cases <- fread("data-raw/covid19-utah-cases.csv")

ntimes <- 60
ids <- seq(1, nrow(utah_cases)-ntimes)
ids <- cbind(ids[-length(ids)], ids[-1])
raw_data <- prepare_data_infections_only.default(
  utah_cases$cases,
  max_days = 60
  )


