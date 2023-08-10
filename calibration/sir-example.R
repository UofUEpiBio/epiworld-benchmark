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
  prob_transmission = truth[3],
  prob_recovery     = truth[4],
  n                 = 2000
)

set.seed(100)
run(abm, 50)

dat_prep <- prepare_data(abm)

a <- array(dim = c(1, dim(dat_prep)))
a[1,,] <- dat_prep
abm_hist_feat <- a

abm_hist_feat <- array_reshape(
  abm_hist_feat,
  dim = c(1, dim(dat_prep))
  )

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
  prob_transmission = truth[3],
  prob_recovery     = truth[4],
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
