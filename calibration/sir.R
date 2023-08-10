# devtools::install_github("UofUEpi/epiworldR")
source("calibration/dataprep.R")

N     <- 2e4
n     <- 2000
ndays <- 50
ncores <- 20

set.seed(1231)

theta <- data.table(
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
        prob_transmission = ptran,
        prob_recovery     = prec,
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

# Setting up the data for tensorflow. Need to figure out how we would configure
# this to store an array of shape 3 x 100 (three rows, S I R) and create the 
# convolution.

# Convolutional Neural Network
library(keras)

# (N obs, rows, cols)
# Important note, it is better for the model to handle changes rather than
# total numbers. For the next step, we need to do it using % change, maybe...
arrays_1d <- array(dim = c(N, dim(matrices[[1]])))
for (i in seq_along(matrices))
  arrays_1d[i,,] <- matrices[[i]]
    #   t(matrices[[i]][-nrow(matrices[[i]]),]) + 1e-20
    # )[,1:49]
    
    # t(diff(t(matrices[[i]])))/(
    #   matrices[[i]][,-ncol(matrices[[i]])] + 1e-20
    # )[,1:50]

theta2 <- copy(theta)
theta2[, crate := plogis(crate)]

# N <- 200L

# Reshaping

N_train <- floor(N * .7)
id_train <- 1:N_train
train <- list(
  x = array_reshape(
    arrays_1d[id_train,,], dim = c(N_train, dim(arrays_1d)[-1])
    ),
  y = array_reshape(
    as.matrix(theta2)[id_train,], dim = c(N_train, ncol(theta2)))
    )


N_test <- N - N_train
id_test <- (N_train + 1):N

test <- list(
  x = array_reshape(arrays_1d[id_test,,], dim = c(N_test, dim(arrays_1d)[-1])),
  y = array_reshape(as.matrix(theta2)[id_test,], dim = c(N_test, ncol(theta2)))
)

# Follow examples in: https://tensorflow.rstudio.com/tutorials/keras/classification

# Build the model
model <- keras_model_sequential()
model %>%
  layer_conv_2d(
    filters     = 32,
    input_shape = c(dim(arrays_1d)[-1], 1),
    activation  = "linear",
    kernel_size = c(3, 5)
    ) %>%
  layer_max_pooling_2d(
    pool_size = 2,
    padding = 'same'
    ) %>%
  layer_flatten(
    input_shape = dim(arrays_1d)[-1]
    ) %>%
  # layer_normalization() %>%
  layer_dense(
    units = ncol(theta2),
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

MAEs <- abs(pred - as.matrix(test$y)) |>
  colMeans() |>
  print()

save_model_hdf5(model, "sir-keras")

# Visualizing ------------------------------------------------------------------
pred[, id := 1L:.N]
pred[, crate := qlogis(crate)]
pred_long <- melt(pred, id.vars = "id")

theta_long <- test$y |> as.data.table()
setnames(theta_long, names(theta))
theta_long[, id := 1L:.N]
theta_long[, crate := qlogis(crate)]
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
  variable = c("preval", "crate", "ptran", "prec"),
  Name     = paste(
    c("Init. state", "Contact Rate", "P(transmit)", "P(recover)"),
    sprintf("(MAE: %.2f)", MAEs)
    )
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

ggsave(filename = "calibration/sir.png", width = 1280, height = 800, units = "px", scale = 3)
