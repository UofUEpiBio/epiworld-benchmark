# Using the LSTM model to Find the Best Parameters Used in Generating the SIR Model

# Section 1:

#### implementing LSTM model

installing required packages:

``` r
library(epiworldR)
library(data.table)
library(tensorflow)
library(keras)
```


    Attaching package: 'keras'

    The following object is masked from 'package:epiworldR':

        clone_model

``` r
library(parallel)
library(keras3)
```

    Registered S3 methods overwritten by 'keras3':
      method                               from 
      as.data.frame.keras_training_history keras
      plot.keras_training_history          keras
      print.keras_training_history         keras
      r_to_py.R6ClassGenerator             keras


    Attaching package: 'keras3'

    The following objects are masked from 'package:keras':

        %<-active%, %py_class%, activation_elu, activation_exponential,
        activation_gelu, activation_hard_sigmoid, activation_linear,
        activation_relu, activation_selu, activation_sigmoid,
        activation_softmax, activation_softplus, activation_softsign,
        activation_tanh, adapt, application_densenet121,
        application_densenet169, application_densenet201,
        application_efficientnet_b0, application_efficientnet_b1,
        application_efficientnet_b2, application_efficientnet_b3,
        application_efficientnet_b4, application_efficientnet_b5,
        application_efficientnet_b6, application_efficientnet_b7,
        application_inception_resnet_v2, application_inception_v3,
        application_mobilenet, application_mobilenet_v2,
        application_mobilenet_v3_large, application_mobilenet_v3_small,
        application_nasnetlarge, application_nasnetmobile,
        application_resnet101, application_resnet101_v2,
        application_resnet152, application_resnet152_v2,
        application_resnet50, application_resnet50_v2, application_vgg16,
        application_vgg19, application_xception, bidirectional,
        callback_backup_and_restore, callback_csv_logger,
        callback_early_stopping, callback_lambda,
        callback_learning_rate_scheduler, callback_model_checkpoint,
        callback_reduce_lr_on_plateau, callback_remote_monitor,
        callback_tensorboard, clone_model, constraint_maxnorm,
        constraint_minmaxnorm, constraint_nonneg, constraint_unitnorm,
        count_params, custom_metric, dataset_boston_housing,
        dataset_cifar10, dataset_cifar100, dataset_fashion_mnist,
        dataset_imdb, dataset_imdb_word_index, dataset_mnist,
        dataset_reuters, dataset_reuters_word_index, freeze_weights,
        from_config, get_config, get_file, get_layer, get_vocabulary,
        get_weights, image_array_save, image_dataset_from_directory,
        image_load, image_to_array, imagenet_decode_predictions,
        imagenet_preprocess_input, initializer_constant,
        initializer_glorot_normal, initializer_glorot_uniform,
        initializer_he_normal, initializer_he_uniform,
        initializer_identity, initializer_lecun_normal,
        initializer_lecun_uniform, initializer_ones,
        initializer_orthogonal, initializer_random_normal,
        initializer_random_uniform, initializer_truncated_normal,
        initializer_variance_scaling, initializer_zeros, install_keras,
        keras, keras_model, keras_model_sequential, Layer,
        layer_activation, layer_activation_elu,
        layer_activation_leaky_relu, layer_activation_parametric_relu,
        layer_activation_relu, layer_activation_softmax,
        layer_activity_regularization, layer_add, layer_additive_attention,
        layer_alpha_dropout, layer_attention, layer_average,
        layer_average_pooling_1d, layer_average_pooling_2d,
        layer_average_pooling_3d, layer_batch_normalization,
        layer_category_encoding, layer_center_crop, layer_concatenate,
        layer_conv_1d, layer_conv_1d_transpose, layer_conv_2d,
        layer_conv_2d_transpose, layer_conv_3d, layer_conv_3d_transpose,
        layer_conv_lstm_1d, layer_conv_lstm_2d, layer_conv_lstm_3d,
        layer_cropping_1d, layer_cropping_2d, layer_cropping_3d,
        layer_dense, layer_depthwise_conv_1d, layer_depthwise_conv_2d,
        layer_discretization, layer_dot, layer_dropout, layer_embedding,
        layer_flatten, layer_gaussian_dropout, layer_gaussian_noise,
        layer_global_average_pooling_1d, layer_global_average_pooling_2d,
        layer_global_average_pooling_3d, layer_global_max_pooling_1d,
        layer_global_max_pooling_2d, layer_global_max_pooling_3d,
        layer_gru, layer_hashing, layer_input, layer_integer_lookup,
        layer_lambda, layer_layer_normalization, layer_lstm, layer_masking,
        layer_max_pooling_1d, layer_max_pooling_2d, layer_max_pooling_3d,
        layer_maximum, layer_minimum, layer_multi_head_attention,
        layer_multiply, layer_normalization, layer_permute,
        layer_random_brightness, layer_random_contrast, layer_random_crop,
        layer_random_flip, layer_random_rotation, layer_random_translation,
        layer_random_zoom, layer_repeat_vector, layer_rescaling,
        layer_reshape, layer_resizing, layer_rnn, layer_separable_conv_1d,
        layer_separable_conv_2d, layer_simple_rnn,
        layer_spatial_dropout_1d, layer_spatial_dropout_2d,
        layer_spatial_dropout_3d, layer_string_lookup, layer_subtract,
        layer_text_vectorization, layer_unit_normalization,
        layer_upsampling_1d, layer_upsampling_2d, layer_upsampling_3d,
        layer_zero_padding_1d, layer_zero_padding_2d,
        layer_zero_padding_3d, learning_rate_schedule_cosine_decay,
        learning_rate_schedule_cosine_decay_restarts,
        learning_rate_schedule_exponential_decay,
        learning_rate_schedule_inverse_time_decay,
        learning_rate_schedule_piecewise_constant_decay,
        learning_rate_schedule_polynomial_decay, loss_binary_crossentropy,
        loss_categorical_crossentropy, loss_categorical_hinge,
        loss_cosine_similarity, loss_hinge, loss_huber, loss_kl_divergence,
        loss_mean_absolute_error, loss_mean_absolute_percentage_error,
        loss_mean_squared_error, loss_mean_squared_logarithmic_error,
        loss_poisson, loss_sparse_categorical_crossentropy,
        loss_squared_hinge, mark_active, metric_auc,
        metric_binary_accuracy, metric_binary_crossentropy,
        metric_categorical_accuracy, metric_categorical_crossentropy,
        metric_categorical_hinge, metric_cosine_similarity,
        metric_false_negatives, metric_false_positives, metric_hinge,
        metric_mean, metric_mean_absolute_error,
        metric_mean_absolute_percentage_error, metric_mean_iou,
        metric_mean_squared_error, metric_mean_squared_logarithmic_error,
        metric_mean_wrapper, metric_poisson, metric_precision,
        metric_precision_at_recall, metric_recall,
        metric_recall_at_precision, metric_root_mean_squared_error,
        metric_sensitivity_at_specificity,
        metric_sparse_categorical_accuracy,
        metric_sparse_categorical_crossentropy,
        metric_sparse_top_k_categorical_accuracy,
        metric_specificity_at_sensitivity, metric_squared_hinge,
        metric_sum, metric_top_k_categorical_accuracy,
        metric_true_negatives, metric_true_positives, new_callback_class,
        new_layer_class, new_learning_rate_schedule_class, new_loss_class,
        new_metric_class, new_model_class, normalize, optimizer_adadelta,
        optimizer_adagrad, optimizer_adam, optimizer_adamax,
        optimizer_ftrl, optimizer_nadam, optimizer_rmsprop, optimizer_sgd,
        pad_sequences, pop_layer, predict_on_batch, regularizer_l1,
        regularizer_l1_l2, regularizer_l2, regularizer_orthogonal,
        set_vocabulary, set_weights, shape, test_on_batch,
        text_dataset_from_directory, time_distributed,
        timeseries_dataset_from_array, to_categorical, train_on_batch,
        unfreeze_weights, use_backend, with_custom_object_scope, zip_lists

    The following objects are masked from 'package:tensorflow':

        set_random_seed, shape

    The following object is masked from 'package:epiworldR':

        clone_model

``` r
library(dplyr)
```


    Attaching package: 'dplyr'

    The following objects are masked from 'package:data.table':

        between, first, last

    The following objects are masked from 'package:stats':

        filter, lag

    The following objects are masked from 'package:base':

        intersect, setdiff, setequal, union

``` r
library(ggplot2)
```

## calling Preparation Function

Now we call the source to prepare the data we want to generate in the
next step.

``` r
# devtools::install_github("UofUEpi/epiworldR")
source("calibration/dataprep.R")
```

# Set parameters

``` r
N     <- 2e4
n     <- 5000
ndays <- 50
ncores <- 20
```

# Generate parameters (theta)

The `generate_simulation_parameters` function generates a table of
parameters (`theta`) for the SIR model simulations.

``` r
set.seed(1231)
generate_simulation_parameters <- function(N, n) {
  theta <- data.table(
    preval = sample((100:2000) / n, N, TRUE),
    crate  = rgamma(N, 5, 1),  # Mean 10
    ptran  = rbeta(N, 3, 7),   # Mean 3/(3 + 7) = 0.3
    prec   = rbeta(N, 10, 10 * 2 - 10)  # Mean 10 / (10 * 2 - 10) = .5
  )
  theta[, hist(crate)]
  return(theta)
}
```

- **Inputs**:

  - `N`: The number of simulations or rows of parameter sets to
    generate.

  - `n`: The population size used for prevalence calculations.

- **Outputs**:

  - The function returns a `data.table` containing four key
    epidemiological parameters:

    1.  **`preval`**: Prevalence, or the initial proportion of the
        population that is infected. It is sampled from a range between
        `100/n` and `2000/n` (which corresponds to infection rates
        between 0.02% and 0.4%).

    2.  **`crate`**: Contact rate, which is the rate at which
        individuals in the population come into contact with others. It
        is generated from a Gamma distribution with a shape parameter of
        5 and a rate parameter of 1 (which gives a mean contact rate of
        5).

    3.  **`ptran`**: Transmission probability, or the likelihood of
        disease transmission upon contact, drawn from a Beta
        distribution with shape parameters 3 and 7. This gives a mean
        transmission rate of 0.3 (since 3 / (3 + 7) = 0.3).

    4.  **`prec`**: Recovery probability, or the likelihood of recovery
        after infection. It is drawn from a Beta distribution with shape
        parameters of 10 and 10, giving a mean recovery rate of 0.5
        (since 10 / (10 \* 2 - 10) = 0.5).

# run epidemic model simulations in parallel

`run_epidemic_simulations` Function runs SIR
(Susceptible-Infectious-Recovered) model simulations in parallel for `N`
iterations, using a set of parameters (`theta`) for each simulation.

``` r
run_epidemic_simulations <- function(N, theta, seeds, n, ndays, ncores) {
  matrices <- parallel::mclapply(1:N, FUN = function(i) {
    fn <- sprintf("calibration/simulated_data/sir-%06i.rds", i)
    if (file.exists(fn)) return(readRDS(fn))
    set.seed(seeds[i])
    m <- theta[i, ModelSIRCONN("mycon", prevalence = preval, contact_rate = crate, transmission_rate = ptran, recovery_rate = prec, n = n)]
    verbose_off(m)
    run(m, ndays = ndays)
    ans <- prepare_data(m)
    saveRDS(ans, fn)
    return(ans)
  }, mc.cores = ncores)
  return(matrices)
}
```

``` r
filter_valid_simulations <- function(matrices, theta) {
  valid_indices <- intersect(
    which(!sapply(matrices, inherits, what = "error")),
    which(!sapply(matrices, \(x) any(is.na(x))))
  )
  matrices <- matrices[valid_indices]
  theta <- theta[valid_indices, ]
  return(list(matrices = matrices, theta = theta, N = length(valid_indices)))
}
```

**Inputs**:

- `N`: Number of simulations to run.

- `theta`: A table containing the parameters for each simulation
  (prevalence, contact rate, transmission rate, and recovery rate).

- `seeds`: A set of random seeds to ensure each simulation has different
  random behavior.

- `n`: Population size for the simulation.

- `ndays`: Number of days to simulate.

- `ncores`: Number of CPU cores to use for parallel processing.

# prepare simulation data for neural network training

`prepare_neural_network_data` Function prepares the simulation data and
parameters for use in training a neural network.

``` r
# Function to prepare simulation data for neural network training
prepare_neural_network_data <- function(N, matrices, theta) {
  arrays_1d <- array(dim = c(N, dim(matrices[[1]][1, , ])))
  for (i in seq_along(matrices)) {
    arrays_1d[i, , ] <- matrices[[i]][1, , ]
  }
  theta2 <- copy(theta)
  theta2[, crate := plogis(crate / 10)]
  return(list(arrays_1d = arrays_1d, theta2 = theta2))
}

# Function to save the prepared data to an RDS file
save_prepared_data <- function(theta2, arrays_1d) {
  saveRDS(
    list(
      theta = theta2,
      simulations = arrays_1d
    ),
    file = "calibration/sir.rds",
    compress = TRUE
  )
}
```

# Split data into training and test sets

``` r
split_train_test_data <- function(N, arrays_1d, theta2) {
  N_train <- floor(N * 0.7)
  id_train <- 1:N_train
  id_test <- (N_train + 1):N
  train <- list(
    x = array_reshape(arrays_1d[id_train, , ], dim = c(N_train, dim(arrays_1d)[-1])),
    y = array_reshape(as.matrix(theta2)[id_train, ], dim = c(N_train, ncol(theta2)))
  )
  test <- list(
    x = array_reshape(arrays_1d[id_test, , ], dim = c(N - N_train, dim(arrays_1d)[-1])),
    y = array_reshape(as.matrix(theta2)[id_test, ], dim = c(N - N_train, ncol(theta2)))
  )
  return(list(train = train, test = test))
}
```

- **Inputs**:

  - `N`: The number of valid simulations.

  - `matrices`: A list of simulation results (from
    `run_epidemic_simulations`).

  - `theta`: A table of parameters used for the simulations.

- **Functionality**:

  - **Creating an Array for Simulations**:

    - `arrays_1d`: A 3D array is created to store the first slice of
      data (e.g., infections) from each simulation result. The
      dimensions are `(N, rows, cols)` where `N` is the number of
      simulations.

    - Each simulation’s result is extracted (`matrices[[i]][1, , ]`) and
      added to `arrays_1d`.

  - **Adjusting Parameters**:

    - `theta2`: A copy of `theta` is created, and the contact rate
      (`crate`) is transformed using the logistic function
      (`plogis(crate / 10)`) to map values into a probability range (0,
      1).

# Build and train the LSTM model

`build_and_train_lstm` Function builds and trains an LSTM (Long
Short-Term Memory) neural network model using the `keras3` package.

``` r
build_and_train_lstm <- function(train, theta2, arrays_1d) {
  model <- keras3::keras_model_sequential() %>%
    keras3::layer_lstm(units = 64, input_shape = c(dim(arrays_1d)[2], dim(arrays_1d)[3]), return_sequences = FALSE) %>%
    keras3::layer_dense(units = ncol(theta2), activation = 'sigmoid')
  
  model %>% compile(optimizer = 'adam', loss = 'mse', metrics = 'accuracy')
  
  tensorflow::set_random_seed(331)
  model %>% fit(train$x, train$y, epochs = 100, verbose = 0)
  
  return(model)
}
```

**Inputs**:

- `train`: The training dataset, consisting of `train$x` (input data)
  and `train$y` (target parameters).

- `theta2`: The adjusted table of parameters that the model will learn
  to predict.

- `arrays_1d`: The 3D array containing simulation data, which helps
  define the input shape of the LSTM.

- `lstm_units`: The number of LSTM units (default is 64) in the hidden
  layer.

- `epochs`: Number of training epochs (default is 100), i.e., how many
  times the model will iterate over the training data.

- `loss_function`: The loss function used to optimize the model (default
  is Mean Squared Error, `'mse'`).

- `optimizer`: The optimizer for training (default is Adam).

- `activation`: The activation function used in the output layer
  (default is `'sigmoid'`).

# predict using the trained model

``` r
# Function to predict using the trained model
make_predictions <- function(model, test, theta) {
  pred <- predict(model, x = test$x) |> as.data.table() |> setnames(colnames(theta))
  return(pred)
}

# Function to compute the Mean Absolute Error (MAE)
compute_mae <- function(pred, test) {
  MAEs <- abs(pred - as.matrix(test$y)) |> colMeans() |> print()
  return(MAEs)
}
```

# Visualize model predictions against observed data

``` r
visualize_predictions <- function(pred, test, MAEs, theta, N, N_train) {
  pred[, id := 1L:.N]
  pred[, crate := qlogis(crate) * 10]
  pred_long <- melt(pred, id.vars = "id")
  
  theta_long <- test$y |> as.data.table()
  setnames(theta_long, names(theta))
  theta_long[, id := 1L:.N]
  theta_long[, crate := qlogis(crate) * 10]
  theta_long <- melt(theta_long, id.vars = "id")
  
  alldat <- rbind(cbind(pred_long, Type = "Predicted"), cbind(theta_long, Type = "Observed"))
  
  ggplot(alldat, aes(x = value, colour = Type)) +
    facet_wrap(~variable, scales = "free") +
    geom_boxplot()
  
  alldat_wide <- dcast(alldat, id + variable ~ Type, value.var = "value")
  
  vnames <- data.table(
    variable = c("preval", "crate", "ptran", "prec"),
    Name     = paste(c("Init. state", "Contact Rate", "P(transmit)", "P(recover)"), sprintf("(MAE: %.2f)", MAEs))
  )
  
  alldat_wide <- merge(alldat_wide, vnames, by = "variable")
  
  ggplot(alldat_wide, aes(x = Observed, y = Predicted)) +
    facet_wrap(~ Name, scales = "free") +
    geom_abline(slope = 1, intercept = 0) +
    geom_point(alpha = .2) +
    labs(
      title = "Observed vs Predicted (validation set)",
      subtitle = sprintf("The model includes %i simulated datasets, of which %i were used for training.", N, N_train),
      caption = "Predictions made using a CNN as implemented with loss function MAE."
    )
}
```

# Main execution block

``` r
theta <- generate_simulation_parameters(N, n)
```

![](LSTM_SIR_all_files/figure-commonmark/run%20the%20single%20LSTM%20model-1.png)

``` r
seeds <- sample.int(.Machine$integer.max, N, TRUE)
matrices <- run_epidemic_simulations(N, theta, seeds, n, ndays, ncores)

# Filter and prepare data
valid_data <- filter_valid_simulations(matrices, theta)
N <- valid_data$N
theta <- valid_data$theta
matrices <- valid_data$matrices

# Prepare for neural network
nn_data <- prepare_neural_network_data(N, matrices, theta)
arrays_1d <- nn_data$arrays_1d
theta2 <- nn_data$theta2

# Save data
save_prepared_data(theta2, arrays_1d)

# Split train and test data
train_test_data <- split_train_test_data(N, arrays_1d, theta2)
train <- train_test_data$train
test <- train_test_data$test
```

``` r
# Build, train, and evaluate the model
model <- build_and_train_lstm(train, theta2, arrays_1d)
```

``` r
pred <- make_predictions(model, test, theta)
```

    188/188 - 0s - 2ms/step

``` r
MAEs <- compute_mae(pred, test)
```

        preval      crate      ptran       prec 
    0.01795201 0.03666196 0.08452936 0.03105621 

``` r
print(MAEs)
```

        preval      crate      ptran       prec 
    0.01795201 0.03666196 0.08452936 0.03105621 

``` r
visualize_predictions(pred, test, MAEs, theta, N, floor(N * 0.7))
```

![](LSTM_SIR_all_files/figure-commonmark/Visualize%20the%20results-1.png)

# Section 2:

# Finding the Best Model

Now we want to find the best model for LSTM.

# build and train the LSTM model with hyperparameters

The `build_and_train_model` function builds, trains, and evaluates an
LSTM (Long Short-Term Memory) neural network using the `keras3` library
in R for time-series or sequential data.

``` r
#|label: Function to build and train the LSTM model with hyperparameters
build_and_train_model <- function(train, test, theta, seed,
                                  units, activation_lstm, activation_dense,
                                  optimizer, loss, epochs, batch_size, verbose = 0) {
  # Build the LSTM model
  model <- keras3::keras_model_sequential() %>%
    keras3::layer_lstm(
      units = units,
      activation = activation_lstm,
      input_shape = c(dim(train$x)[2], dim(train$x)[3]),
      return_sequences = FALSE
    ) %>%
    keras3::layer_dense(
      units = ncol(theta),
      activation = activation_dense
    )
  
  # Compile the model
  model %>% keras3::compile(
    optimizer = optimizer,
    loss      = loss,
    metrics   = 'mae'
  )
  
  # Set random seed
  tensorflow::set_random_seed(seed)
  
  # Fit the model
  history <- model %>% keras3::fit(
    x = train$x,
    y = train$y,
    epochs = epochs,
    batch_size = batch_size,
    validation_split = 0.2,
    verbose = verbose
  )
  
  # Make predictions
  pred <- predict(model, x = test$x) %>%
    as.data.table() %>%
    setnames(colnames(theta))
  
  # Calculate MAEs
  MAEs <- abs(pred - test$y) %>%
    colMeans()
  
  # Return the MAEs and predictions
  list(pred = pred, MAEs = MAEs, model = model)
}
```

### **Building the LSTM Model**

- The function creates a sequential model using
  `keras3::keras_model_sequential()`.

- An LSTM layer (`keras3::layer_lstm()`) is added with customizable
  parameters:

  - `units`: the number of LSTM units (neurons) in the layer.

  - `activation_lstm`: the activation function applied within the LSTM
    units.

  - `input_shape`: the shape of the input data, which is defined as the
    number of time steps (`dim(train$x)[2]`) and features
    (`dim(train$x)[3]`).

  - `return_sequences = FALSE`: this flag means that only the final LSTM
    output (not a sequence) is passed to the next layer.

- A fully connected (dense) layer (`keras3::layer_dense()`) is added
  with:

  - `units = ncol(theta)`: the number of output neurons, which matches
    the number of columns in the `theta` matrix.

  - `activation_dense`: the activation function applied in the dense
    layer (e.g., ‘sigmoid’ or ‘linear’).

# visualize results

``` r
visualize_results <- function(pred, test, theta, MAEs, N, N_train, output_file = NULL) {
  pred[, id := 1L:.N]
  pred[, crate := qlogis(crate) * 10]
  pred_long <- melt(pred, id.vars = "id")
  
  theta_long <- as.data.table(test$y)
  setnames(theta_long, names(theta))
  theta_long[, id := 1L:.N]
  theta_long[, crate := qlogis(crate) * 10]
  theta_long <- melt(theta_long, id.vars = "id")
  
  alldat <- rbind(
    cbind(pred_long, Type = "Predicted"),
    cbind(theta_long, Type = "Observed")
  )
  
  # Density plots
  p1 <- ggplot(alldat, aes(x = value, colour = Type)) +
    facet_wrap(~variable, scales = "free") +
    geom_density() +
    labs(title = "Density Plots of Predicted vs Observed Values",
         subtitle = "Comparing distributions of predicted and observed parameters",
         x = "Parameter Value", y = "Density")
  
  print(p1)
  
  # Scatter plots of Observed vs Predicted
  alldat_wide <- dcast(alldat, id + variable ~ Type, value.var = "value")
  
  vnames <- data.table(
    variable = c("preval", "crate", "ptran", "prec"),
    Name     = paste(
      c("Init. state", "Contact Rate", "P(transmit)", "P(recover)"),
      sprintf("(MAE: %.4f)", MAEs)
    )
  )
  
  alldat_wide <- merge(alldat_wide, vnames, by = "variable")
  
  p2 <- ggplot(alldat_wide, aes(x = Observed, y = Predicted)) +
    facet_wrap(~ Name, scales = "free") +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
    geom_point(alpha = .2) +
    labs(
      title    = "Observed vs Predicted (Test Set)",
      subtitle = sprintf(
        "Best Model with Mean MAE: %.4f",
        mean(MAEs)
      ),
      x = "Observed Values",
      y = "Predicted Values"
    )
  
  print(p2)
}
```

# Execution function with hyperparameter tuning

This `main` function performs a full workflow to generate and train an
LSTM-based neural network model using hyperparameter tuning.

``` r
main <- function(N, n, ndays, ncores) {
  # Generate parameters (theta)
  theta <- generate_simulation_parameters(N, n)
  seeds <- sample.int(.Machine$integer.max, N, TRUE)
  
  # Run simulations
  matrices <- run_epidemic_simulations(N, theta, seeds, n, ndays, ncores)
  
  # Filter and prepare data
  valid_data <- filter_valid_simulations(matrices, theta)
  N <- valid_data$N
  theta <- valid_data$theta
  matrices <- valid_data$matrices
  
  # Prepare for neural network
  nn_data <- prepare_neural_network_data(N, matrices, theta)
  arrays_1d <- nn_data$arrays_1d
  theta2 <- nn_data$theta2
  
  # Save data
  save_prepared_data(theta2, arrays_1d)
  
  # Split train and test data
  train_test_data <- split_train_test_data(N, arrays_1d, theta2)
  train <- train_test_data$train
  test <- train_test_data$test
  N_train <- train_test_data$N_train
  
  # Define hyperparameter grid
  hyper_grid <- expand.grid(
    units = c(50, 64),
    activation_lstm = c('relu'),
    activation_dense = c('sigmoid',"linear"),
    optimizer = c('adam'),
    loss = c('mse', 'mae'),
    epochs = c(20),
    batch_size = c(32, 64),
    stringsAsFactors = FALSE
  )
  
  # Initialize variables to store the best model
  best_MAE <- Inf
  best_model <- NULL
  best_pred <- NULL
  best_MAEs <- NULL
  best_params <- NULL
  
  # Loop over hyperparameter combinations
  for (i in 1:nrow(hyper_grid)) {
    cat("Testing model", i, "of", nrow(hyper_grid), "\n")
    
    # Extract hyperparameters
    units <- hyper_grid$units[i]
    activation_lstm <- hyper_grid$activation_lstm[i]
    activation_dense <- hyper_grid$activation_dense[i]
    optimizer <- hyper_grid$optimizer[i]
    loss <- hyper_grid$loss[i]
    epochs <- hyper_grid$epochs[i]
    batch_size <- hyper_grid$batch_size[i]
    
    # Set a seed for reproducibility
    seed <- 331
    
    # Build and train the model
    model_results <- tryCatch(
      {
        build_and_train_model(
          train = train,
          test = test,
          theta = theta2,
          seed = seed,
          units = units,
          activation_lstm = activation_lstm,
          activation_dense = activation_dense,
          optimizer = optimizer,
          loss = loss,
          epochs = epochs,
          batch_size = batch_size,
          verbose = 0
        )
      },
      error = function(e) {
        cat("Error in model", i, ":", e$message, "\n")
        return(NULL)
      }
    )
    
    # If the model failed, skip to the next iteration
    if (is.null(model_results)) next
    
    # Get the MAEs
    MAEs <- model_results$MAEs
    
    # Store the average MAE
    hyper_grid$MAE[i] <- mean(MAEs)
    
    # If this is the best MAE so far, save the model and predictions
    if (hyper_grid$MAE[i] < best_MAE) {
      best_MAE <- hyper_grid$MAE[i]
      best_model <- model_results$model
      best_pred <- model_results$pred
      best_MAEs <- MAEs
      best_params <- hyper_grid[i,]
    }
  }
  
  # Print the best hyperparameters
  cat("Best model parameters:\n")
  print(best_params)
  cat("Best MAE:", best_MAE, "\n")
  
  # Visualize results
  visualize_results(best_pred, test, theta2, best_MAEs, N, N_train)
}
```

### 1. **Parameter Generation**

- `generate_simulation_parameters(N, n)`: This function generates
  simulation parameters (`theta`) for the epidemic simulations based on
  the input parameters `N` and `n`. The exact nature of `theta` depends
  on the simulation context.

- `sample.int(.Machine$integer.max, N, TRUE)`: This generates a set of
  random seeds for reproducibility across `N` simulations.

### 2. **Run Simulations**

- `run_epidemic_simulations(N, theta, seeds, n, ndays, ncores)`: This
  runs the epidemic simulations using the parameters (`theta`) and seeds
  across `ncores` processing cores. The results are stored in
  `matrices`, which likely contain simulated data for different
  scenarios.

### 3. **Filter and Prepare Data**

- `filter_valid_simulations(matrices, theta)`: This function filters out
  invalid or null simulations. It adjusts `matrices`, `theta`, and `N`
  to ensure only valid simulations are used for further processing.

- `prepare_neural_network_data(N, matrices, theta)`: Prepares the data
  for input into the LSTM neural network, converting the simulation
  results (`matrices`) and parameters (`theta`) into appropriate arrays
  (`arrays_1d`) for TensorFlow.

### 4. **Save Prepared Data**

- `save_prepared_data(theta2, arrays_1d)`: Saves the prepared data
  (optional), possibly as `.rds` or some other serialized format,
  allowing you to reload it later.

### 5. **Split Train and Test Data**

- `split_train_test_data(N, arrays_1d, theta2)`: Splits the dataset into
  training and test sets. `train_test_data` includes `train`, `test`,
  and the number of training samples (`N_train`).

### 6. **Define Hyperparameter Grid**

- `expand.grid()`: Creates a grid of hyperparameters for tuning. The
  parameters include:

  - `units`: Number of LSTM units (50 or 64).

  - `activation_lstm`: Activation function for LSTM (e.g., `'relu'`).

  - `activation_dense`: Activation function for the dense output layer
    (`'sigmoid'` or `'linear'`).

  - `optimizer`: Optimizer used for training (`'adam'`).

  - `loss`: Loss function for training (`'mse'` or `'mae'`).

  - `epochs`: Number of training epochs (20).

  - `batch_size`: Batch size for training (32 or 64).

This grid allows for different combinations of hyperparameters to be
tested and tuned for the best model performance.

### 7. **Hyperparameter Tuning Loop**

- The loop runs through all possible hyperparameter combinations:

  - Extracts the current hyperparameters from the grid (`units`,
    `activation_lstm`, `activation_dense`, `optimizer`, `loss`,
    `epochs`, `batch_size`).

  - Sets a reproducible seed (`seed = 331`).

  - Calls the `build_and_train_model` function to build, train, and test
    the model for each combination of hyperparameters.

### 8. **Error Handling**

- A `tryCatch` block is used to handle errors during model building or
  training. If an error occurs, it moves to the next combination of
  hyperparameters without stopping the entire process.

### 9. **Evaluating the Models**

- After each model is trained, the mean absolute errors (MAEs) are
  calculated.

- The average MAE is stored for each hyperparameter configuration.

- If the current model has the best MAE (lowest), it updates the stored
  `best_model`, `best_pred`, `best_MAEs`, and `best_params`.

### 10. **Results**

- After testing all hyperparameter combinations, the function prints the
  best model’s hyperparameters and the lowest MAE achieved.

### 11. **Visualization**

- `visualize_results()`: This function visualizes the results, such as
  comparing predicted values (`best_pred`) to the actual values
  (`test$y`). It likely includes error analysis, plotting, and model
  diagnostics.

# Run the main function

``` r
#|label: Run the main function
N <- 2e4   # Adjust N as needed
n <- 5000
ndays <- 50
ncores <- 20
main(N, n, ndays, ncores)
```

![](LSTM_SIR_all_files/figure-commonmark/unnamed-chunk-22-1.png)

    Testing model 1 of 16 
    188/188 - 0s - 2ms/step
    Testing model 2 of 16 
    188/188 - 0s - 2ms/step
    Testing model 3 of 16 
    188/188 - 0s - 2ms/step
    Testing model 4 of 16 
    188/188 - 0s - 2ms/step
    Testing model 5 of 16 
    188/188 - 0s - 2ms/step
    Testing model 6 of 16 
    188/188 - 0s - 2ms/step
    Testing model 7 of 16 
    188/188 - 0s - 2ms/step
    Testing model 8 of 16 
    188/188 - 0s - 2ms/step
    Testing model 9 of 16 
    188/188 - 0s - 2ms/step
    Testing model 10 of 16 
    188/188 - 0s - 2ms/step
    Testing model 11 of 16 
    188/188 - 0s - 2ms/step
    Testing model 12 of 16 
    188/188 - 0s - 2ms/step
    Testing model 13 of 16 
    188/188 - 0s - 2ms/step
    Testing model 14 of 16 
    188/188 - 0s - 2ms/step
    Testing model 15 of 16 
    188/188 - 0s - 2ms/step
    Testing model 16 of 16 
    188/188 - 0s - 2ms/step
    Best model parameters:
      units activation_lstm activation_dense optimizer loss epochs batch_size
    6    64            relu          sigmoid      adam  mae     20         32
            MAE
    6 0.0845276
    Best MAE: 0.0845276 

![](LSTM_SIR_all_files/figure-commonmark/unnamed-chunk-22-2.png)

![](LSTM_SIR_all_files/figure-commonmark/unnamed-chunk-22-3.png)
