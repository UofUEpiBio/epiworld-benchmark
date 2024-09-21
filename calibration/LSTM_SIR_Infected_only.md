# Implementing the LSTM Model Just by Knowing the Infections Count to
Find the Closest Parameters to Simulate SIR Models

Load required libraries

``` r
library(data.table)
library(parallel)
library(keras)
library(ggplot2)
library(reshape2)
```


    Attaching package: 'reshape2'

    The following objects are masked from 'package:data.table':

        dcast, melt

``` r
library(tensorflow)
library(keras3)
```

    Registered S3 methods overwritten by 'keras3':
      method                               from 
      as.data.frame.keras_training_history keras
      plot.keras_training_history          keras
      print.keras_training_history         keras
      r_to_py.R6ClassGenerator             keras


    Attaching package: 'keras3'

    The following objects are masked from 'package:tensorflow':

        set_random_seed, shape

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

# simulate data

This simulate_data function simulates epidemiological data based on the
SIR (Susceptible-Infected-Recovered) model across multiple iterations
and saves the results.

``` r
simulate_data <- function(N = 2e4, n = 5000, ndays = 50, ncores = 20, seed = 1231, savefile = "calibration/sir.rds") {
  source("calibration/dataprep.R")
  
  set.seed(seed)
  
  theta <- data.table(
    preval = sample((100:2000)/n, N, TRUE),
    crate  = rgamma(N, 5, 1),    # Mean 5
    ptran  = rbeta(N, 3, 7),     # Mean 0.3
    prec   = rbeta(N, 10, 10*2 - 10) # Mean 0.5
  )
  
  seeds <- sample.int(.Machine$integer.max, N, TRUE)
  
  matrices <- parallel::mclapply(1:N, FUN = function(i) {
    fn <- sprintf("calibration/simulated_data/sir-%06i.rds", i)
    
    if (file.exists(fn))
      return(readRDS(fn))
    
    set.seed(seeds[i])
    
    m <- theta[i,
               ModelSIRCONN(
                 "mycon",
                 prevalence        = preval,
                 contact_rate      = crate,
                 transmission_rate = ptran,
                 recovery_rate     = prec, 
                 n                 = n
               )
    ]
    
    # Avoids printing
    verbose_off(m)
    
    run(m, ndays = ndays)
    
    # Using prepare_data
    ans <- prepare_data(m)
    saveRDS(ans, fn)
    
    ans
  }, mc.cores = ncores)
  
  # Keeping only the non-null elements
  is_not_null <- intersect(
    which(!sapply(matrices, inherits, what = "error")),
    which(!sapply(matrices, function(x) any(is.na(x))))
  )
  matrices <- matrices[is_not_null]
  theta    <- theta[is_not_null,]
  
  N <- length(is_not_null)
  
  # Setting up the data for tensorflow
  arrays_1d <- array(dim = c(N, dim(matrices[[1]][1,,])))
  for (i in seq_along(matrices))
    arrays_1d[i,,] <- matrices[[i]][1,,]
  
  theta2 <- copy(theta)
  theta2[, crate := plogis(crate / 10)]
  
  # Saving the data 
  saveRDS(
    list(
      theta = theta2,
      simulations = arrays_1d
    ),
    file = savefile,
    compress = TRUE
  )
}
```

### **Parameters**:

- `N`: Number of simulations to run (default 20,000).

- `n`: Population size in each simulation (default 5,000).

- `ndays`: Number of days to simulate (default 50).

- `ncores`: Number of CPU cores used for parallel processing (default
  20).

- `seed`: Random seed for reproducibility (default 1231).

- `savefile`: The file path to save the output (default
  `"calibration/sir.rds"`).

# prepare training and testing data

This function prepares simulation data for training machine learning
models. It splits the data into training and test sets based on the
specified training fraction, reshapes the data, and returns it in a
format ready for the LSTM model.

``` r
#|label: Function to prepare training and testing data
prepare_data_sets <- function(datafile = "calibration/sir.rds", train_fraction = 0.7, ndays = 50) {
  sim_results <- readRDS(datafile)
  theta <- sim_results$theta
  arrays_1d <- sim_results$simulations
  
  # Extracting infections only
  arrays_1d <- arrays_1d[,1,,drop=FALSE] # dimensions: (N, 1, ndays)
  N     <- dim(arrays_1d)[1]
  
  # Reshaping for LSTM input: (samples, timesteps, features)
  N_train <- floor(N * train_fraction)
  id_train <- 1:N_train
  
  # Ensure dimensions are consistent and reshape
  train_x <- arrays_1d[id_train,1,, drop = FALSE] # dimensions: (N_train, 1, ndays)
  train_x <- aperm(train_x, c(1, 3, 2)) # dimensions: (N_train, ndays, 1)
  
  train_y <- as.matrix(theta)[id_train,]
  
  N_test <- N - N_train
  id_test <- (N_train + 1):N
  
  test_x <- arrays_1d[id_test,1,, drop = FALSE] # dimensions: (N_test, 1, ndays)
  test_x <- aperm(test_x, c(1, 3, 2)) # dimensions: (N_test, ndays, 1)
  
  test_y <- as.matrix(theta)[id_test,]
  
  list(
    train = list(x = train_x, y = train_y),
    test = list(x = test_x, y = test_y),
    theta = theta,
    arrays_1d = arrays_1d,
    N = N,
    N_train = N_train,
    N_test = N_test,
    ndays = ndays
  )
}
```

# build and train the model

The `build_and_train_model` function builds, trains, and evaluates a
LSTM model using the `keras3` and `tensorflow` R packages.

``` r
#|label: Function to build and train the LSTM model
build_and_train_model <- function(train, test, theta, N_train, ndays, seed = 331, save_model_file = "sir-lstm_infections_only") {
  # Build the LSTM model
  model <- keras_model_sequential() %>%
    layer_lstm(
      units = 50,
      input_shape = c(ndays, 1)
    ) %>%
    layer_dense(
      units = ncol(theta),
      activation = 'sigmoid'
    )
  
  # Compile the model
  model %>% compile(
    optimizer = 'adam',
    loss      = 'mse',
    metrics    = 'mae'
  )
  
  # Running the model
  tensorflow::set_random_seed(seed)
  history <- model %>% fit(
    x = train$x,
    y = train$y,
    epochs = 50,
    batch_size = 64,
    validation_split = 0.2,
    verbose = 0
  )
  
  pred <- predict(model, x = test$x) %>%
    as.data.table() %>%
    setnames(colnames(theta))
  
  MAEs <- abs(pred - as.matrix(test$y)) %>%
    colMeans() %>%
    print()
  
  # Save the model

  list(pred = pred, MAEs = MAEs, history = history)
}
```

### **Parameters**:

- `train`: The training data, including inputs (`x`) and targets (`y`).

- `test`: The test data for model evaluation.

- `arrays_1d`: The 1D simulation data is used to define the input shape
  for the model.

- `theta`: The target variables from the simulations.

- `N_train`: Number of training samples.

- `seed`: A random seed for reproducibility (default: 331).

- `save_model_file`: Filename to save the model (currently unused in the
  code).

# visualize results

The `visualize_results` function generates visualizations to compare
predicted values from a trained model with the test dataset’s observed
(true) values.

``` r
visualize_results <- function(pred, test, theta, MAEs, N, N_train, output_file = "calibration/sir_infections_only.png") {
  pred[, id := 1L:.N]
  pred[, crate := qlogis(crate)]
  pred_long <- melt(pred, id.vars = "id")
  
  theta_long <- test$y %>% as.data.table()
  setnames(theta_long, names(theta))
  theta_long[, id := 1L:.N]
  theta_long[, crate := qlogis(crate)]
  theta_long <- melt(theta_long, id.vars = "id")
  
  alldat <- rbind(
    cbind(pred_long, Type = "Predicted"),
    cbind(theta_long, Type = "Observed")
  )
  
  # Boxplot
  p1 <- ggplot(alldat, aes(x = value, colour = Type)) +
    facet_wrap(~variable, scales = "free") +
    geom_boxplot()
  print(p1)
  
  alldat_wide <- dcast(alldat, id + variable ~ Type, value.var = "value")
  
  vnames <- data.table(
    variable = c("preval", "crate", "ptran", "prec"),
    Name     = paste(
      c("Init. state", "Contact Rate", "P(transmit)", "P(recover)"),
      sprintf("(MAE: %.2f)", MAEs)
    )
  )
  
  alldat_wide <- merge(alldat_wide, vnames, by = "variable")
  
  # Scatter plot
  p2 <- ggplot(alldat_wide, aes(x = Observed, y = Predicted)) +
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
      caption  = "Predictions made using an LSTM with loss function MSE."
    )
  print(p2)
  
  # Save the plot
  ggsave(filename = output_file, width = 1280, height = 800, units = "px", scale = 3)
}
```

# Running the model

``` r
#|label: Main execution function
main <- function() {
  # Simulate data
  simulate_data()
  
  # Prepare data sets
  data_sets <- prepare_data_sets()
  train <- data_sets$train
  test <- data_sets$test
  theta <- data_sets$theta
  N <- data_sets$N
  N_train <- data_sets$N_train
  ndays <- data_sets$ndays
  
  # Build and train the LSTM model
  model_results <- build_and_train_model(train, test, theta, N_train, ndays)
  pred <- model_results$pred
  MAEs <- model_results$MAEs
  
  # Visualize results
  visualize_results(pred, test, theta, MAEs, N, N_train)
}

# Run the main function
main()
```


    Attaching package: 'epiworldR'

    The following object is masked from 'package:keras3':

        clone_model

    The following object is masked from 'package:keras':

        clone_model

    188/188 - 1s - 4ms/step
        preval      crate      ptran       prec 
    0.02364865 0.03144636 0.07145539 0.05783265 

![](LSTM_SIR_Infected_only_files/figure-commonmark/unnamed-chunk-9-1.png)

![](LSTM_SIR_Infected_only_files/figure-commonmark/unnamed-chunk-9-2.png)

# Section 2

# Finding the Best Model

Now we want to find the best model for LSTM.

# build and train the LSTM model with hyperparameters

The `build_and_train_model` function builds, trains, and evaluates an
LSTM (Long Short-Term Memory) neural network using the `keras3` library
in R for time-series or sequential data.

``` r
build_and_train_model <- function(train, test, theta, N_train, ndays, seed,
                                  units, activation_lstm, activation_dense,
                                  optimizer, loss, epochs, batch_size, verbose = 0) {
  # Build the LSTM model
  model <- keras_model_sequential() %>%
    layer_lstm(
      units = units,
      activation = activation_lstm,
      input_shape = c(ndays, 1)
    ) %>%
    layer_dense(
      units = ncol(theta),
      activation = activation_dense
    )
  
  # Compile the model
  model %>% compile(
    optimizer = optimizer,
    loss      = loss,
    metrics   = 'mae'
  )
  
  # Set random seed
  tensorflow::set_random_seed(seed)
  
  # Fit the model
  history <- model %>% fit(
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
#|label: Function to visualize results
visualize_results <- function(pred, test, theta, MAEs, N, N_train, output_file = NULL) {
  pred[, id := 1L:.N]
  pred_long <- melt(pred, id.vars = "id")
  
  theta_long <- as.data.table(test$y)
  setnames(theta_long, names(theta))
  theta_long[, id := 1L:.N]
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
    variable = names(theta),
    Name     = paste(
      c("Initial Prevalence", "Contact Rate", "Transmission Probability", "Recovery Probability"),
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
main <- function() {
  # Simulate data
  simulate_data()
  
  # Prepare data sets
  data_sets <- prepare_data_sets()
  train <- data_sets$train
  test <- data_sets$test
  theta <- data_sets$theta
  N <- data_sets$N
  N_train <- data_sets$N_train
  ndays <- data_sets$ndays
  
  # Define hyperparameter grid
  hyper_grid <- expand.grid(
    units = c(50, 64),
    activation_lstm = c( 'relu'),
    activation_dense = c('sigmoid', 'linear'),
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
          theta = theta,
          N_train = N_train,
          ndays = ndays,
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
  visualize_results(best_pred, test, theta, best_MAEs, N, N_train)
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
main()
```

    Testing model 1 of 16 
    188/188 - 1s - 4ms/step
    Testing model 2 of 16 
    188/188 - 1s - 5ms/step
    Testing model 3 of 16 
    188/188 - 1s - 4ms/step
    Testing model 4 of 16 
    188/188 - 1s - 5ms/step
    Testing model 5 of 16 
    188/188 - 1s - 5ms/step
    Testing model 6 of 16 
    188/188 - 1s - 4ms/step
    Testing model 7 of 16 
    188/188 - 1s - 4ms/step
    Testing model 8 of 16 
    188/188 - 1s - 4ms/step
    Testing model 9 of 16 
    188/188 - 1s - 4ms/step
    Testing model 10 of 16 
    188/188 - 1s - 5ms/step
    Testing model 11 of 16 
    188/188 - 1s - 4ms/step
    Testing model 12 of 16 
    188/188 - 1s - 5ms/step
    Testing model 13 of 16 
    188/188 - 1s - 4ms/step
    Testing model 14 of 16 
    188/188 - 1s - 4ms/step
    Testing model 15 of 16 
    188/188 - 1s - 4ms/step
    Testing model 16 of 16 
    188/188 - 1s - 4ms/step
    Best model parameters:
      units activation_lstm activation_dense optimizer loss epochs batch_size
    5    50            relu          sigmoid      adam  mae     20         32
             MAE
    5 0.04730755
    Best MAE: 0.04730755 

![](LSTM_SIR_Infected_only_files/figure-commonmark/unnamed-chunk-16-1.png)

![](LSTM_SIR_Infected_only_files/figure-commonmark/unnamed-chunk-16-2.png)
