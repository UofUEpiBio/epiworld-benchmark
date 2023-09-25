library(keras)
library(epiworldR)
source("calibration/dataprep.R")
saved_model <- load_model_hdf5("sir-keras")

set.seed(33331)
popsize <- 2000

# Step 1: Simulate an SEIR model for T days
seir_model <- ModelSEIRCONN(
  "myseir",
  prevalence        = .1,
  contact_rate      = 2,
  transmission_rate = .7,
  recovery_rate     = .6,
  incubation_days   = 7,
  n                 = popsize
)

# Running the model
run(seir_model, 100)

history <- as.data.table(get_hist_total(seir_model))

# Step 2: Calibrate the model looking up to T/2 days
abm_hist_feat <- prepare_data(seir_model)

obspars    <- predict(saved_model, x = abm_hist_feat)
obspars[2] <- qlogis(obspars[2])

# Step 3: Simulate the model for T days
abm <- ModelSIRCONN(
  "mycon",
  prevalence        = obspars[1],
  contact_rate      = obspars[2],
  transmission_rate = obspars[3],
  recovery_rate     = obspars[4],
  n                 = popsize
)

run_multiple(
  abm, 100, 1000,
  nthreads = 4,
  saver = make_saver(
    "total_hist", "reproductive",
    fn = "prediction/saves/%03lu"
    )
  )

# Step 4: Compare the results
abm_1000   <- run_multiple_get_results(abm)
post25 <- abm_1000$total_hist[abm_1000$total_hist$date >= 50,]
post25 <- as.data.table(post25)

# Computing the quantiles by date and state
post25 <- post25[, .(
  q025 = quantile(counts, .025),
  q50  = quantile(counts, .50),
  q75  = quantile(counts, .975)
  ), by = .(date, state)]

# Reshape the data
post25 <- melt(post25, id.vars = c("date", "state"))

setnames(history, "counts", "value")
history[, variable := "Observed"]

combined <- rbind(post25, history)

library(ggplot2)
ggplot(combined, aes(x = date, y = value)) +
  geom_line(aes(group = variable, color = variable)) +
  facet_wrap(~state, scales = "free") 
