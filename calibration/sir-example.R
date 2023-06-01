library(keras)
library(epiworldR)
library(data.table)

https://tensorflow.rstudio.com/tutorials/keras/save_and_load.html
saved_model <- load_model_hdf5("sir-keras")

truth <- c(.1, 2, .7, .6)

# Simulating a SIR model
abm <- ModelSIRCONN(
  "mycon",
  prevalence          = truth[1],
  reproductive_number = truth[2],
  prob_transmission   = truth[3],
  prob_recovery       = truth[4],
  n                   = 2000
)

set.seed(100)
run(abm, 50)

abm_hist <- get_hist_total(abm)
abm_hist_feat <- matrix(abm_hist$counts, nrow = 3)
abm_hist_feat <- t(diff(t(abm_hist_feat)))[,1:50]

a <- array(dim = c(1, 3, 50))
a[1,,] <- abm_hist_feat
abm_hist_feat <- a

abm_hist_feat <- array_reshape(
  abm_hist_feat,
  dim = c(1, 3, 50)
  )

obspars <- predict(saved_model, x = abm_hist_feat)
obspars[2] <- qlogis(obspars[2])

res <- data.table(
  Parameter = c("Init. state", "Beta (repnum)", "P(transmit)", "P(recover)"),
  Predicted = round(obspars[1,], 2),
  Truth     = truth
)

# knitr::kable(res, format = "html")

plot(abm)


run_multiple(abm, 50, 1000, nthreads = 4, saver = make_saver("total_hist"), reset = TRUE)
abm_1000 <- run_multiple_get_results(abm)$total_hist
abm_1000 <- abm_1000[abm_1000$date <= 20,]

library(ggplot2)
ggplot(abm_1000, aes(group = date, y = counts)) +
  facet_wrap(~state, scales = "free") + 
  geom_boxplot()
