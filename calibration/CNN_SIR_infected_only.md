# Implementing the CNN Model Just by Knowing the Infections Count to
Find the Closest Parameters to Simulate SIR Models

Load required libraries:

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

`imulate_data` function performs a simulation for epidemiological
modeling using the SIR (Susceptible-Infectious-Recovered) model across
multiple iterations, to generate and save data for further analysis.

``` r
#|label: Function to simulate data
simulate_data <- function(N = 2e4, n = 5000, ndays = 50, ncores = 20, seed = 1231, savefile = "calibration/sir.rds") {
  source("calibration/dataprep.R")
  
  set.seed(seed)
  
  theta <- data.table(
    preval = sample((100:2000)/n, N, TRUE),
    crate  = rgamma(N, 5, 1),    # Mean 5
    ptran  = rbeta(N, 3, 7),         # Mean 0.3
    prec   = rbeta(N, 10, 10*2 - 10) # Mean 0.5
  )
  
  theta[, hist(crate)]
  
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

**Parameters**:

- `N`: Number of simulations to run (default 20,000).

- `n`: Population size for each simulation (default 5,000).

- `ndays`: Number of days to run each simulation (default 50).

- `ncores`: Number of CPU cores used for parallel processing (default
  20).

- `seed`: Random seed for reproducibility.

- `savefile`: Path to save the final output.

# prepare training and testing data

This function prepares simulation data for training machine learning
models. It splits the data into training and test sets based on the
specified training fraction, reshapes the data, and returns it in a
format ready for the CNN model.

``` r
prepare_data_sets <- function(datafile = "calibration/sir.rds", train_fraction = 0.7) {
  sim_results <- readRDS(datafile)
  theta <- sim_results$theta
  arrays_1d <- sim_results$simulations
  
  # Extracting infections only
  arrays_1d <- arrays_1d[,1,,drop=FALSE]
  N     <- dim(arrays_1d)[1]
  
  # Reshaping
  N_train <- floor(N * train_fraction)
  id_train <- 1:N_train
  train <- list(
    x = array_reshape(
      arrays_1d[id_train,,], dim = c(N_train, dim(arrays_1d)[-1])
    ),
    y =  array_reshape(
      as.matrix(theta)[id_train,], dim = c(N_train, ncol(theta)))
  )
  
  N_test <- N - N_train
  id_test <- (N_train + 1):N
  
  test <- list(
    x = array_reshape(arrays_1d[id_test,,], dim = c(N_test, dim(arrays_1d)[-1])),
    y = array_reshape(as.matrix(theta)[id_test,], dim = c(N_test, ncol(theta)))
  )
  
  list(train = train, test = test, theta = theta, arrays_1d = arrays_1d, N = N, N_train = N_train, N_test = N_test)
}
```

# build and train the model

The `build_and_train_model` function builds, trains, and evaluates a
convolutional neural network (CNN) model using the `keras3` and
`tensorflow` R packages.

``` r
#|label: Function to build and train the model
build_and_train_model <- function(train, test, arrays_1d, theta, N_train, seed = 331, save_model_file = "sir-keras_infections_only") {
  # Build the model
  model <- keras3::keras_model_sequential()
  model |>
    keras3::layer_conv_2d(
      filters     = 32,
      input_shape = c(dim(arrays_1d)[-1], 1),
      activation  = "linear",
      kernel_size = c(1, 5)
    ) |>
    keras3::layer_max_pooling_2d(
      pool_size = 2,
      padding = 'same'
    ) |>
    keras3::layer_flatten(
      input_shape = dim(arrays_1d)[-1]
    ) |>
    keras3::layer_dense(
      units = ncol(theta),
      activation = 'sigmoid'
    )
  
  # Compile the model
  model %>% compile(
    optimizer = 'adam',
    loss      = 'mse',
    metric    = 'accuracy'
  )
  
  # Running the model
  tensorflow::set_random_seed(seed)
  model |> fit(
    train$x,
    train$y,
    epochs = 50,
    verbose = 0
  )
  
  pred <- predict(model, x = test$x) |>
    as.data.table() |>
    setnames(colnames(theta))
  
  MAEs <- abs(pred - as.matrix(test$y)) |>
    colMeans() |>
    print()
  

  list(pred = pred, MAEs = MAEs)
}
```

### **Parameters**:

- `train`: The training data, including inputs (`x`) and targets (`y`).

- `test`: The test data for model evaluation.

- `arrays_1d`: The 1D simulation data used to define the input shape for
  the model.

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
#|label: Function to visualize results
visualize_results <- function(pred, test, theta, MAEs, N, N_train, output_file = "calibration/sir_infections_only.png") {
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
  
  p1<-ggplot(alldat, aes(x = value, colour = Type)) +
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
  
  p2<-ggplot(alldat_wide, aes(x = Observed, y = Predicted)) +
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
      caption  = "Predictions made using a CNN(infections only) as implemented with loss function MAE."
    )
  print(p2)
  
 
}
```

# Running the Model

``` r
main <- function() {
  # Simulate data
  simulate_data()
  
  # Prepare data sets
  data_sets <- prepare_data_sets()
  train <- data_sets$train
  test <- data_sets$test
  theta <- data_sets$theta
  arrays_1d <- data_sets$arrays_1d
  N <- data_sets$N
  N_train <- data_sets$N_train
  
  # Build and train the model
  model_results <- build_and_train_model(train, test, arrays_1d, theta, N_train)
  pred <- model_results$pred
  MAEs <- model_results$MAEs
  
  # Visualize results
  print(visualize_results(pred, test, theta, MAEs, N, N_train))
}

# Run the main function
main()
```


    Attaching package: 'epiworldR'

    The following object is masked from 'package:keras3':

        clone_model

    The following object is masked from 'package:keras':

        clone_model

![](CNN_SIR_infected_only_files/figure-commonmark/Main%20execution%20function-1.png)

    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.03639818 0.03265657 0.07754360 0.08081101 

![](CNN_SIR_infected_only_files/figure-commonmark/Main%20execution%20function-2.png)

![](CNN_SIR_infected_only_files/figure-commonmark/Main%20execution%20function-3.png)

![](CNN_SIR_infected_only_files/figure-commonmark/Main%20execution%20function-4.png)

# Section 2:

# Finding the Best CNN model

``` r
build_and_train_model <- function(train, test, theta, seed,
                                  filters, kernel_size, activation_conv,
                                  activation_dense, pool_size, optimizer,
                                  loss, epochs, verbose = 0) {
  # Build the model
  model <- keras::keras_model_sequential()
  model %>%
    keras::layer_conv_2d(
      filters     = filters,
      input_shape = c(dim(train$x)[2], dim(train$x)[3], 1),
      activation  = activation_conv,
      kernel_size = kernel_size
    ) %>%
    keras::layer_max_pooling_2d(
      pool_size = pool_size,
      padding = 'same'
    ) %>%
    keras::layer_flatten() %>%
    keras::layer_dense(
      units = ncol(theta),
      activation = activation_dense
    )
  
  # Compile the model
  model %>% keras::compile(
    optimizer = optimizer,
    loss      = loss,
    metrics   = 'mae'
  )
  
  # Set random seed
  tensorflow::set_random_seed(seed)
  
  # Fit the model
  model %>% keras::fit(
    train$x,
    train$y,
    epochs = epochs,
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

# Function to visualize results
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

# Main execution function with hyperparameter tuning
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
  
  # Reshape the data for Keras
  train$x <- array(train$x, dim = c(dim(train$x)[1], dim(train$x)[2], dim(train$x)[3], 1))
  test$x <- array(test$x, dim = c(dim(test$x)[1], dim(test$x)[2], dim(test$x)[3], 1))
  
  # Define hyperparameter grid
  hyper_grid <- expand.grid(
    filters = c(16, 32, 64),
    kernel_size = list(c(1,3), c(1,5)),
    activation_conv = c('relu', 'linear'),
    activation_dense = c('sigmoid'),
    pool_size = list(c(1,2), c(1,3)),
    optimizer = c('adam'),
    loss = c('mse', 'mae'),
    epochs = c(50),
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
    filters <- hyper_grid$filters[i]
    kernel_size <- hyper_grid$kernel_size[[i]]
    activation_conv <- hyper_grid$activation_conv[i]
    activation_dense <- hyper_grid$activation_dense[i]
    pool_size <- hyper_grid$pool_size[[i]]
    optimizer <- hyper_grid$optimizer[i]
    loss <- hyper_grid$loss[i]
    epochs <- hyper_grid$epochs[i]
    
    # Set a seed for reproducibility
    seed <- 331
    
    # Build and train the model
    model_results <- tryCatch(
      {
        build_and_train_model(
          train = train,
          test = test,
          theta = theta,
          seed = seed,
          filters = filters,
          kernel_size = kernel_size,
          activation_conv = activation_conv,
          activation_dense = activation_dense,
          pool_size = pool_size,
          optimizer = optimizer,
          loss = loss,
          epochs = epochs,
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

# Run the main function
main()
```

![](CNN_SIR_infected_only_files/figure-commonmark/unnamed-chunk-11-1.png)

    Testing model 1 of 48 
    188/188 - 0s - 1ms/step
    Testing model 2 of 48 
    188/188 - 0s - 1ms/step
    Testing model 3 of 48 
    188/188 - 0s - 1ms/step
    Testing model 4 of 48 
    188/188 - 0s - 1ms/step
    Testing model 5 of 48 
    188/188 - 0s - 1ms/step
    Testing model 6 of 48 
    188/188 - 0s - 1ms/step
    Testing model 7 of 48 
    188/188 - 0s - 1000us/step
    Testing model 8 of 48 
    188/188 - 0s - 1ms/step
    Testing model 9 of 48 
    188/188 - 0s - 1ms/step
    Testing model 10 of 48 
    188/188 - 0s - 1ms/step
    Testing model 11 of 48 
    188/188 - 0s - 1ms/step
    Testing model 12 of 48 
    188/188 - 0s - 1ms/step
    Testing model 13 of 48 
    188/188 - 0s - 1ms/step
    Testing model 14 of 48 
    188/188 - 0s - 1ms/step
    Testing model 15 of 48 
    188/188 - 0s - 1ms/step
    Testing model 16 of 48 
    188/188 - 0s - 1ms/step
    Testing model 17 of 48 
    188/188 - 0s - 1ms/step
    Testing model 18 of 48 
    188/188 - 0s - 1ms/step
    Testing model 19 of 48 
    188/188 - 0s - 1ms/step
    Testing model 20 of 48 
    188/188 - 0s - 1ms/step
    Testing model 21 of 48 
    188/188 - 0s - 1ms/step
    Testing model 22 of 48 
    188/188 - 0s - 1ms/step
    Testing model 23 of 48 
    188/188 - 0s - 1ms/step
    Testing model 24 of 48 
    188/188 - 0s - 1ms/step
    Testing model 25 of 48 
    188/188 - 0s - 1ms/step
    Testing model 26 of 48 
    188/188 - 0s - 1ms/step
    Testing model 27 of 48 
    188/188 - 0s - 1ms/step
    Testing model 28 of 48 
    188/188 - 0s - 1ms/step
    Testing model 29 of 48 
    188/188 - 0s - 1ms/step
    Testing model 30 of 48 
    188/188 - 0s - 1ms/step
    Testing model 31 of 48 
    188/188 - 0s - 1ms/step
    Testing model 32 of 48 
    188/188 - 0s - 997us/step
    Testing model 33 of 48 
    188/188 - 0s - 1ms/step
    Testing model 34 of 48 
    188/188 - 0s - 1ms/step
    Testing model 35 of 48 
    188/188 - 0s - 1ms/step
    Testing model 36 of 48 
    188/188 - 0s - 1ms/step
    Testing model 37 of 48 
    188/188 - 0s - 1ms/step
    Testing model 38 of 48 
    188/188 - 0s - 1ms/step
    Testing model 39 of 48 
    188/188 - 0s - 1ms/step
    Testing model 40 of 48 
    188/188 - 0s - 1ms/step
    Testing model 41 of 48 
    188/188 - 0s - 1ms/step
    Testing model 42 of 48 
    188/188 - 0s - 1ms/step
    Testing model 43 of 48 
    188/188 - 0s - 1ms/step
    Testing model 44 of 48 
    188/188 - 0s - 996us/step
    Testing model 45 of 48 
    188/188 - 0s - 1ms/step
    Testing model 46 of 48 
    188/188 - 0s - 1ms/step
    Testing model 47 of 48 
    188/188 - 0s - 1ms/step
    Testing model 48 of 48 
    188/188 - 0s - 1ms/step
    Best model parameters:
       filters kernel_size activation_conv activation_dense pool_size optimizer
    39      64        1, 3            relu          sigmoid      1, 3      adam
       loss epochs        MAE
    39  mae     50 0.05267334
    Best MAE: 0.05267334 

![](CNN_SIR_infected_only_files/figure-commonmark/unnamed-chunk-11-2.png)

![](CNN_SIR_infected_only_files/figure-commonmark/unnamed-chunk-11-3.png)
