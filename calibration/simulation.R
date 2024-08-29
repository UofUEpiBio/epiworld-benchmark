library(keras)
library(epiworldR)
source("calibration/dataprep.R")
saved_model <- load_model_hdf5("sir-keras")

set.seed(33331)
popsize <- 10000
ndays   <- 100

# Step 1: Simulate an SEIR model for T days
seir_model <- ModelSEIRCONN(
  "myseir",
  prevalence        = 0.1,
  contact_rate      = 2,
  transmission_rate = .5,
  recovery_rate     = .6,
  incubation_days   = 7,
  n                 = popsize
)

# Running the model
run(seir_model, ndays)

history <- as.data.table(get_hist_total(seir_model))

# Step 2: Calibrate the model looking up to T/2 days
abm_hist_feat <- prepare_data(seir_model, max_days = ndays/2)

obspars    <- predict(saved_model, x = abm_hist_feat)
obspars[2] <- qlogis(obspars[2])

# Step 3: Simulate the model for T days
abm <- ModelSIRCONN(
  "mycon",
  prevalence        = history[date == ndays/2 & state == "Infected", counts]/popsize,
  contact_rate      = obspars[2],
  transmission_rate = obspars[3],
  recovery_rate     = obspars[4],
  n                 = popsize
)

# Need to mimic the initial condition of number of susceptibles, 
# so we distribute a tool with 100% immunity
immu <- tool(
  "immunity", susceptibility_reduction = 1, transmission_reduction = 0,
  recovery_enhancer = 0, death_reduction = 0
  )

add_tool(
  abm, immu,
  proportion = history[date == ndays & state == "Recovered", counts]/popsize
  )

# Running 200 replicates of the model
if (!dir.exists("prediction/saves"))
  dir.create("prediction/saves")

run_multiple(
  abm, ndays/2, 200,
  nthreads = 4,
  saver = make_saver(
    "total_hist",
    "reproductive",
    fn = "prediction/saves/%03lu"
    )
  )

# Step 4: Compare the results
abm_1000   <- run_multiple_get_results(abm)
post25 <- as.data.table(abm_1000$total_hist)[date < ndays/2]
post25[, date := date + ndays/2]


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
