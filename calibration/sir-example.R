library(keras)
source("calibration/dataprep.R")

# https://tensorflow.rstudio.com/tutorials/keras/save_and_load.html
saved_model <- load_model_hdf5("sir-keras")

truth <- c(.1, 2, .7, .6)

# Simulating a SIR model
abm <- ModelSIRCONN(
  "mycon",
  prevalence        = truth[1],
  contact_rate      = truth[2],
  transmission_rate = truth[3],
  recovery_rate     = truth[4],
  n                 = 2000
)

set.seed(100)
run(abm, 50)

# Sets up the data so we can pass it to the model
abm_hist_feat <- prepare_data(abm)

obspars <- predict(saved_model, x = abm_hist_feat)
obspars[2] <- qlogis(obspars[2])

res <- data.table(
  Parameter = c("Init. state", "Contact Rate", "P(transmit)", "P(recover)"),
  Predicted = round(obspars[1,], 2),
  Truth     = truth
)

# knitr::kable(res, format = "html")
knitr::kable(res)

plot(abm)

set.seed(123)
abm <- ModelSIRCONN(
  "mycon",
  prevalence        = truth[1],
  contact_rate      = truth[2],
  transmission_rate = truth[3],
  recovery_rate     = truth[4],
  n                 = 2000
)

run_multiple(
  abm, 50, 1000, nthreads = 4,
  saver = make_saver("total_hist", "reproductive")
  )

abm_1000 <- run_multiple_get_results(abm)
ggplotdata <- abm_1000$total_hist[abm_1000$total_hist$date <= 20,]

library(ggplot2)
ggplot(ggplotdata, aes(group = date, y = counts, x = date)) +
  facet_wrap(~state, scales = "free") + 
  geom_boxplot()

ggsave(filename = "calibration/sir-example.png", width = 1280/2, height = 500/2, units = "px", scale = 3)

plot(abm_1000$reproductive)
