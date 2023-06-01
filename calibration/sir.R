# devtools::install_github("UofUEpi/epiworldR")
library(epiworldR)
library(data.table)

N     <- 1e4
n     <- 2000
ndays <- 50

set.seed(1231)

theta <- data.table(
  preval = rbeta(N, 1, 19),        # Mean 10/(10 + 190) = 0.05
  crate  = rgamma(N, 4, 4/1.5),    # Mean 4/(4 + 1.5) = 1.5
  ptran  = rbeta(N, 7, 3),        # Mean 7/(3 + 7) = 0.7
  prec   = rbeta(N, 10, 10*2 - 10) # Mean 10 / (10 * 2 - 10) = .5
)
theta[, hist(crate)]

ans <- vector("list", N)
for (i in 1:N) {
  
  m <- theta[i,
    ModelSIRCONN(
      "mycon",
      prevalence        = preval,
      contact_rate      = crate,
      prob_transmission = ptran,
      prob_recovery     = prec,
      n                 = n
      )
    ]
  
  # Avoids printing
  verbose_off(m)
  
  run(m, ndays = ndays)
  ans[[i]] <- list(
    repnum = plot(get_reproductive_number(m), plot = FALSE),
    incidence = plot(get_hist_transition_matrix(m), plot = FALSE),
    gentime = plot(get_generation_time(m), plot = FALSE)
  )

  # Filling
  ans[[i]] <- lapply(ans[[i]], as.data.table)

  # Replacing NaN and NAs with the previous value
  # in each element in the list
  ans[[i]]$repnum[, avg := nafill(avg, "locf"), by = .(variant)]
  ans[[i]]$gentime[, gentime_avg := nafill(gentime_avg, "locf"), by = .(virus_id)]
  
  if (!i %% 100) 
    message("Model ", i, " done.")

  # stop()

}

# Setting up the data for tensorflow. Need to figure out how we would configure
# this to store an array of shape 3 x 100 (three rows, S I R) and create the 
# convolution.

# 100 Matrices of 3x101 - SIR as rows 
matrices <- parallel::mclapply(
  X   = ans,
  FUN = function(x) {
    
    susceptible <- x[x[,2] == "Susceptible",]$counts
    infected    <- x[x[,2] == "Infected",]$counts
    recovered   <- x[x[,2] == "Recovered",]$counts
    
    matrix(rbind(susceptible, infected, recovered), nrow = 3)
    
  },
  mc.cores = 4L
  )

# Convolutional Neural Network
library(keras)

# (N obs, rows, cols)
# Important note, it is better for the model to handle changes rather than
# total numbers. For the next step, we need to do it using % change, maybe...
arrays_1d <- array(dim = c(N, 3, 50))
for (i in seq_along(matrices))
  arrays_1d[i,,] <-
    t(diff(t(matrices[[i]])))[,1:50]
    # t(diff(t(matrices[[i]])))/(
    #   matrices[[i]][,-ncol(matrices[[i]])] + 1e-20
    # )[,1:50]

theta2 <- copy(theta)
theta2[, repnum := plogis(repnum)]

# N <- 200L

# Reshaping
N_train <- floor(N * .7)
id_train <- 1:N_train
train <- list(
  x = array_reshape(arrays_1d[id_train,,], dim = c(N_train, 3, 50)),
  y = array_reshape(as.matrix(theta2)[id_train,], dim = c(N_train, 4))
)

N_test <- N - N_train
id_test <- (N_train + 1):N

test <- list(
  x = array_reshape(arrays_1d[id_test,,], dim = c(N_test, 3, 50)),
  y = array_reshape(as.matrix(theta2)[id_test,], dim = c(N_test, 4))
)

# Follow examples in: https://tensorflow.rstudio.com/tutorials/keras/classification

# Build the model
model <- keras_model_sequential()
model %>%
  layer_conv_2d(
    filters     = 32,
    input_shape = c(3, 50, 1),
    activation  = "linear",
    kernel_size = c(3, 5)
    ) %>%
  layer_max_pooling_2d(
    pool_size = 2,
    padding = 'same'
    ) %>%
  layer_flatten(
    input_shape = c(3, 50)
    ) %>%
  # layer_normalization() %>%
  layer_dense(
    units = 4,
    activation = 'sigmoid'
    )

# Compile the model
model %>% compile(
  optimizer = 'adam',
  loss      = 'mse',
  metric    = 'accuracy'
)

# Running the model
tensorflow::set_random_seed(331)
model %>% fit(
  train$x,
  train$y,
  epochs = 100,
  verbose = 2
  )

pred <- predict(model, x = test$x) |>
  as.data.table() |>
  setnames(colnames(theta))

abs(pred - as.matrix(test$y)) |>
  colMeans()

save_model_hdf5(model, "sir-keras")

# Visualizing ------------------------------------------------------------------
pred[, id := 1L:.N]
pred[, repnum := qlogis(repnum)]
pred_long <- melt(pred, id.vars = "id")

theta_long <- test$y |> as.data.table()
setnames(theta_long, names(theta))
theta_long[, id := 1L:.N]
theta_long[, repnum := qlogis(repnum)]
theta_long <- melt(theta_long, id.vars = "id")

alldat <- rbind(
  cbind(pred_long, Type = "Predicted"),
  cbind(theta_long, Type = "Observed")
)

library(ggplot2)
ggplot(alldat, aes(x = value, colour = Type)) +
  facet_wrap(~variable, scales = "free") +
  geom_boxplot()

alldat_wide <- dcast(alldat, id + variable ~ Type, value.var = "value")

vnames <- data.table(
  variable = c("preval", "repnum", "ptran", "prec"),
  Name     = c("Init. state", "Beta (repnum)", "P(transmit)", "P(recover)")
)

alldat_wide <- merge(alldat_wide, vnames, by = "variable")

ggplot(alldat_wide, aes(x = Observed, y = Predicted)) +
  facet_wrap(~ Name, scales = "free") +
  geom_abline(slope = 1, intercept = 0) +
  geom_point(alpha = .2) +
  labs(
    title    = "Observed vs Predicted (validation set)",
    subtitle = sprintf(
      "The model includes %i simulated datasets, of which %i were used for training.",
      N,
      N_train
      ),
    caption  = "Predictions made using a CNN as implemented with loss function MAE."
    
    )

ggsave(filename = "sir.png", width = 1280, height = 800, units = "px", scale = 3)
