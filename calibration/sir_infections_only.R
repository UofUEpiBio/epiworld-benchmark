# devtools::install_github("UofUEpi/epiworldR")
source("calibration/dataprep.R")

n     <- 2000
ndays <- 50
ncores <- 20

# Retrieving simulation results
sim_results <- readRDS("calibration/sir.rds")

theta2 <- sim_results$theta
arrays_1d <- sim_results$simulations

# Extracting infections only
arrays_1d <- arrays_1d[,1,,drop=FALSE]
N     <- dim(arrays_1d)[1]

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
model |>
  layer_conv_2d(
    filters     = 32,
    input_shape = c(dim(arrays_1d)[-1], 1),
    activation  = "linear",
    kernel_size = c(1, 5)
    ) |>
  layer_max_pooling_2d(
    pool_size = 2,
    padding = 'same'
    ) |>
  layer_flatten(
    input_shape = dim(arrays_1d)[-1]
    ) |>
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
model |> fit(
  train$x,
  train$y,
  epochs = 100,
  verbose = 2
  )

pred <- predict(model, x = test$x) |>
  as.data.table() |>
  setnames(colnames(theta2))

MAEs <- abs(pred - as.matrix(test$y)) |>
  colMeans() |>
  print()

save_model_hdf5(model, "sir-keras_infections_only")

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
