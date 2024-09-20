# Implementing the CNN Model by Knowing SIR Counts to Find the Closest
Parameters to Simulate SIR Models

Installing Packages if necessary:


    Attaching package: 'keras'

    The following object is masked from 'package:epiworldR':

        clone_model

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


    Attaching package: 'dplyr'

    The following objects are masked from 'package:data.table':

        between, first, last

    The following objects are masked from 'package:stats':

        filter, lag

    The following objects are masked from 'package:base':

        intersect, setdiff, setequal, union

## calling Preparation Function

Now we call the source to prepare the data we want to generate in the
next step.

``` r
source("calibration/dataprep.R")
```

#### Generating parameters to Generate the Dataset

This function creates theta values which are needed parameters to
generate the SIR dataset.

Preval, contact, recovery, and transmission rates are generated from
distributions.

``` r
generate_theta <- function(N, n) {
  set.seed(1231)
  theta <- data.table(
    preval = sample((100:2000) / n, N, TRUE),
    crate  = rgamma(N, 5, 1),    # Mean 10
    ptran  = rbeta(N, 3, 7),     # Mean 3/(3 + 7) = 0.3
    prec   = rbeta(N, 10, 10*2 - 10) # Mean 10 / (10 * 2 - 10) = 0.5
  )
  return(theta)
}
```

### Generating the SIR Dataset

here a function simulates the SIR dataset with the EpiwolrdR package and
with the values that are generated in the previous step.

``` r
run_simulations <- function(N, n, ndays, ncores, theta, seeds) {
  matrices <- parallel::mclapply(1:N, FUN = function(i) {
    fn <- sprintf("calibration/simulated_data/sir-%06i.rds", i)
    
    if (file.exists(fn))
      return(readRDS(fn))
    
    set.seed(seeds[i])
    m <- theta[i, ModelSIRCONN(
      "mycon",
      prevalence        = preval,
      contact_rate      = crate,
      transmission_rate = ptran,
      recovery_rate     = prec, 
      n                 = n
    )]
    
    verbose_off(m)
    run(m, ndays = ndays)
    ans <- prepare_data(m)
    saveRDS(ans, fn)
    
    return(ans)
  }, mc.cores = ncores)
  
  return(matrices)
}

# Function to filter non-null matrices and update theta
filter_non_null <- function(matrices, theta) {
  is_not_null <- intersect(
    which(!sapply(matrices, inherits, what = "error")),
    which(!sapply(matrices, \(x) any(is.na(x))))
  )
  
  matrices <- matrices[is_not_null]
  theta    <- theta[is_not_null, ]
  
  return(list(matrices = matrices, theta = theta, N = length(is_not_null)))
}
```

# Section 1

### preparing the Dataset to be Ready for CNN

The function converts a list of matrices into a 3D array by extracting
the first slice from each matrix and stacking them along a new
dimension. This prepares the data for input into a TensorFlow model,
which often requires data in a specific shape.

``` r
prepare_data_for_tensorflow <- function(matrices, N) {
  arrays_1d <- array(dim = c(N, dim(matrices[[1]][1,,])))
  
  for (i in seq_along(matrices)) {
    arrays_1d[i,,] <- matrices[[i]][1,,]
  }
  
  return(arrays_1d)
}
```

# Function to split data into training and test sets

``` r
split_data <- function(arrays_1d, theta2, N) {
  N_train <- floor(N * 0.7)
  id_train <- 1:N_train
  id_test <- (N_train + 1):N
  
  train <- list(
    x = array_reshape(arrays_1d[id_train,,], dim = c(N_train, dim(arrays_1d)[-1])),
    y = array_reshape(as.matrix(theta2)[id_train,], dim = c(N_train, ncol(theta2)))
  )
  
  test <- list(
    x = array_reshape(arrays_1d[id_test,,], dim = c(N - N_train, dim(arrays_1d)[-1])),
    y = array_reshape(as.matrix(theta2)[id_test,], dim = c(N - N_train, ncol(theta2)))
  )
  
  return(list(train = train, test = test))
}
```

# Building the CNN

This function constructs a CNN model using the `keras3` package in R.
The model is intended for regression tasks (since it uses ‘mse’ loss and
‘sigmoid’ activation in the output layer) and consists of convolutional,
pooling, flattening, and dense layers.

``` r
build_cnn_model <- function(input_shape, output_units) {
  model <- keras3::keras_model_sequential() %>%
    keras3::layer_conv_2d(
      filters = 32,
      input_shape = c(input_shape, 1),
      activation = "linear",
      kernel_size = c(3, 5)
    ) %>%
    keras3::layer_max_pooling_2d(pool_size = 2, padding = 'same') %>%
    keras3::layer_flatten(input_shape = input_shape) %>%
    keras3::layer_dense(units = output_units, activation = 'sigmoid')
  
  model %>% compile(optimizer = 'adam', loss = 'mse', metric = 'accuracy')
  return(model)
}
```

step 1:

**Parameters**:

- filters=32: The number of convolution filters (kernels) to use.

- **``` input``_shape``= c(input_shape, 1) ```**: Specifies the shape of
  the input data, which `1` represents a single channel (e.g., grayscale
  images).

- **`activation = "linear"`**: Uses the linear activation function (no
  activation is applied).

- **`kernel_size = c(3, 5)`**: The dimensions of the convolution window
  (3 rows by 5 columns).

step 2

1.  **`layer_max_pooling_2d`**: Adds a max pooling layer to reduce the
    spatial dimensions of the output from the previous layer.

- **Parameters**:

  - **`pool_size = 2`**: The size of the pooling window (2x2).

  - **`padding = 'same'`**: Pads the input so that the output has the
    same dimensions as the input.

  step 3

- **`layer_flatten`**: Flattens the input into a one-dimensional vector.

- **`input_shape = input_shape`**: Specifies the input shape for the
  flatten layer (may be redundant if input shape is already defined).

  step 4

- **`layer_dense`**: Adds a fully connected layer to the model.

- **Parameters**:

  - **`units = output_units`**: The number of neurons in the layer,
    matching the number of output variables.

  - **`activation = 'sigmoid'`**: Uses the sigmoid activation function,
    which outputs values between 0 and 1.

# Function to train the model

``` r
train_model <- function(model, train_data, epochs = 100) {
  tensorflow::set_random_seed(331)
  model %>% fit(train_data$x, train_data$y, epochs = epochs, verbose = 2)
}
```

# Evaluation Function:

This function evaluates the trained model on the test data, makes
predictions, calculates the Mean Absolute Error (MAE) for each output
variable, and returns the predictions along with the MAE values.

``` r
evaluate_model <- function(model, test_data, theta) {
  pred <- predict(model, x = test_data$x) |>
    as.data.table() |>
    setnames(colnames(theta))
  
  MAEs <- abs(pred - as.matrix(test_data$y)) |> colMeans() |> print()
  
  return(list(pred = pred, MAEs = MAEs))
}
```

# Function to plot the results

``` r
plot_results <- function(pred, test_data, theta, MAEs, N, N_train) {
  # Prepare the data for plotting
  pred[, id := 1L:.N]
  pred[, crate := qlogis(crate) * 10]
  pred_long <- melt(pred, id.vars = "id")
  
  theta_long <- test_data$y |> as.data.table()
  setnames(theta_long, names(theta))
  theta_long[, id := 1L:.N]
  theta_long[, crate := qlogis(crate) * 10]
  theta_long <- melt(theta_long, id.vars = "id")
  
  alldat <- rbind(
    cbind(pred_long, Type = "Predicted"),
    cbind(theta_long, Type = "Observed")
  )
  
  # Plot 1: Boxplot of Predicted vs Observed values
  p1 <- ggplot(alldat, aes(x = value, colour = Type)) +
    facet_wrap(~variable, scales = "free") +
    geom_boxplot() +
    labs(title = "Boxplot: Predicted vs Observed")
  
  print(p1)  # Display the first plot
  
  # Prepare data for second plot
  alldat_wide <- dcast(alldat, id + variable ~ Type, value.var = "value")
  
  vnames <- data.table(
    variable = c("preval", "crate", "ptran", "prec"),
    Name     = paste(
      c("Init. state", "Contact Rate", "P(transmit)", "P(recover)"),
      sprintf("(MAE: %.2f)", MAEs)
    )
  )
  
  alldat_wide <- merge(alldat_wide, vnames, by = "variable")
  
  # Plot 2: Observed vs Predicted with MAE labels
  p2 <- ggplot(alldat_wide, aes(x = Observed, y = Predicted)) +
    facet_wrap(~ Name, scales = "free") +
    geom_abline(slope = 1, intercept = 0) +
    geom_point(alpha = .2) +
    labs(
      title    = "Observed vs Predicted (validation set)",
      subtitle = sprintf(
        "The model includes %i simulated datasets, of which %i were used for training.",
        N, N_train
      ),
      caption  = "Predictions made using a CNN as implemented with loss function MAE."
    )
  
  print(p2)  # Display the second plot
}

# Call the function to show the plots
# plot_results(pred, test, theta, MAEs, N, N_train)
```

# The main function to orchestrate the entire process

``` r
main_pipeline <- function(N,n,ndays,ncores) {
  
  # Generate theta and seeds
  theta <- generate_theta(N, n)
  seeds <- sample.int(.Machine$integer.max, N, TRUE)
  
  # Run simulations
  matrices <- run_simulations(N, n, ndays, ncores, theta, seeds)
  
  # Filter non-null elements
  filtered_data <- filter_non_null(matrices, theta)
  matrices <- filtered_data$matrices
  theta <- filtered_data$theta
  N <- filtered_data$N
  
  # Prepare data for TensorFlow
  arrays_1d <- prepare_data_for_tensorflow(matrices, N)
  
  # Save theta and simulations data
  theta2 <- copy(theta)
  theta2[, crate := plogis(crate / 10)]
  saveRDS(list(theta = theta2, simulations = arrays_1d), file = "calibration/sir.rds", compress = TRUE)
  
  # Split data into training and testing sets
  data_split <- split_data(arrays_1d, theta2, N)
  train <- data_split$train
  test <- data_split$test
  
  # Build and train the CNN model
  model <- build_cnn_model(dim(arrays_1d)[-1], ncol(theta2))
  train_model(model, train)
  
  # Evaluate the model
  eval_results <- evaluate_model(model, test, theta)
  pred <- eval_results$pred
  MAEs <- eval_results$MAEs
  
  # Plot the results
  plot_results(pred, test, theta, MAEs, N, floor(N * 0.7))
}
```

The `main_pipeline` function automates the process of:

- **Data Generation**: Creates synthetic parameters and seeds for
  simulation.

- **Simulation Execution**: Runs simulations to generate data.

- **Data Preprocessing**: Filters and formats data for modeling.

- **Model Training**: Builds and trains a CNN model on the data.

- **Model Evaluation**: Assesses model performance using test data.

- **Result Visualization**: Provides graphical insights into model
  predictions and errors.

# Run the whole process with given values

``` r
N <- 2e4
n <- 5000
ndays <- 50
ncores <- 20
# Execute the main pipeline
main_pipeline(N,n,ndays,ncores)
```

    Epoch 1/100
    438/438 - 1s - 3ms/step - accuracy: 0.7245 - loss: 0.0579
    Epoch 2/100
    438/438 - 1s - 1ms/step - accuracy: 0.7501 - loss: 0.0303
    Epoch 3/100
    438/438 - 1s - 1ms/step - accuracy: 0.8114 - loss: 0.0071
    Epoch 4/100
    438/438 - 1s - 1ms/step - accuracy: 0.8412 - loss: 0.0052
    Epoch 5/100
    438/438 - 1s - 1ms/step - accuracy: 0.8591 - loss: 0.0044
    Epoch 6/100
    438/438 - 1s - 1ms/step - accuracy: 0.8706 - loss: 0.0040
    Epoch 7/100
    438/438 - 1s - 1ms/step - accuracy: 0.8783 - loss: 0.0038
    Epoch 8/100
    438/438 - 1s - 1ms/step - accuracy: 0.8847 - loss: 0.0036
    Epoch 9/100
    438/438 - 1s - 1ms/step - accuracy: 0.8840 - loss: 0.0035
    Epoch 10/100
    438/438 - 1s - 1ms/step - accuracy: 0.8866 - loss: 0.0034
    Epoch 11/100
    438/438 - 1s - 1ms/step - accuracy: 0.8888 - loss: 0.0033
    Epoch 12/100
    438/438 - 1s - 1ms/step - accuracy: 0.8929 - loss: 0.0033
    Epoch 13/100
    438/438 - 1s - 1ms/step - accuracy: 0.8936 - loss: 0.0033
    Epoch 14/100
    438/438 - 1s - 1ms/step - accuracy: 0.8952 - loss: 0.0032
    Epoch 15/100
    438/438 - 1s - 1ms/step - accuracy: 0.8956 - loss: 0.0032
    Epoch 16/100
    438/438 - 1s - 1ms/step - accuracy: 0.8964 - loss: 0.0032
    Epoch 17/100
    438/438 - 1s - 1ms/step - accuracy: 0.8982 - loss: 0.0032
    Epoch 18/100
    438/438 - 1s - 1ms/step - accuracy: 0.8996 - loss: 0.0032
    Epoch 19/100
    438/438 - 1s - 1ms/step - accuracy: 0.9000 - loss: 0.0032
    Epoch 20/100
    438/438 - 1s - 1ms/step - accuracy: 0.9018 - loss: 0.0031
    Epoch 21/100
    438/438 - 1s - 1ms/step - accuracy: 0.9010 - loss: 0.0031
    Epoch 22/100
    438/438 - 1s - 1ms/step - accuracy: 0.9005 - loss: 0.0031
    Epoch 23/100
    438/438 - 1s - 1ms/step - accuracy: 0.9014 - loss: 0.0031
    Epoch 24/100
    438/438 - 1s - 1ms/step - accuracy: 0.9027 - loss: 0.0031
    Epoch 25/100
    438/438 - 1s - 1ms/step - accuracy: 0.9011 - loss: 0.0031
    Epoch 26/100
    438/438 - 1s - 1ms/step - accuracy: 0.9020 - loss: 0.0031
    Epoch 27/100
    438/438 - 1s - 1ms/step - accuracy: 0.9020 - loss: 0.0031
    Epoch 28/100
    438/438 - 1s - 1ms/step - accuracy: 0.9024 - loss: 0.0031
    Epoch 29/100
    438/438 - 1s - 1ms/step - accuracy: 0.9018 - loss: 0.0031
    Epoch 30/100
    438/438 - 1s - 1ms/step - accuracy: 0.8997 - loss: 0.0031
    Epoch 31/100
    438/438 - 1s - 1ms/step - accuracy: 0.9030 - loss: 0.0031
    Epoch 32/100
    438/438 - 1s - 1ms/step - accuracy: 0.9024 - loss: 0.0031
    Epoch 33/100
    438/438 - 1s - 1ms/step - accuracy: 0.9025 - loss: 0.0031
    Epoch 34/100
    438/438 - 1s - 1ms/step - accuracy: 0.9022 - loss: 0.0031
    Epoch 35/100
    438/438 - 1s - 1ms/step - accuracy: 0.9030 - loss: 0.0031
    Epoch 36/100
    438/438 - 1s - 1ms/step - accuracy: 0.9025 - loss: 0.0031
    Epoch 37/100
    438/438 - 1s - 1ms/step - accuracy: 0.9022 - loss: 0.0031
    Epoch 38/100
    438/438 - 1s - 1ms/step - accuracy: 0.9008 - loss: 0.0031
    Epoch 39/100
    438/438 - 1s - 1ms/step - accuracy: 0.9037 - loss: 0.0031
    Epoch 40/100
    438/438 - 1s - 1ms/step - accuracy: 0.9021 - loss: 0.0031
    Epoch 41/100
    438/438 - 1s - 1ms/step - accuracy: 0.9012 - loss: 0.0031
    Epoch 42/100
    438/438 - 1s - 1ms/step - accuracy: 0.9015 - loss: 0.0031
    Epoch 43/100
    438/438 - 1s - 1ms/step - accuracy: 0.9010 - loss: 0.0031
    Epoch 44/100
    438/438 - 1s - 1ms/step - accuracy: 0.9022 - loss: 0.0031
    Epoch 45/100
    438/438 - 1s - 1ms/step - accuracy: 0.9004 - loss: 0.0031
    Epoch 46/100
    438/438 - 1s - 1ms/step - accuracy: 0.9020 - loss: 0.0031
    Epoch 47/100
    438/438 - 1s - 1ms/step - accuracy: 0.9025 - loss: 0.0030
    Epoch 48/100
    438/438 - 1s - 1ms/step - accuracy: 0.9007 - loss: 0.0031
    Epoch 49/100
    438/438 - 1s - 1ms/step - accuracy: 0.9025 - loss: 0.0030
    Epoch 50/100
    438/438 - 1s - 1ms/step - accuracy: 0.9010 - loss: 0.0031
    Epoch 51/100
    438/438 - 1s - 1ms/step - accuracy: 0.9020 - loss: 0.0030
    Epoch 52/100
    438/438 - 1s - 1ms/step - accuracy: 0.9027 - loss: 0.0030
    Epoch 53/100
    438/438 - 1s - 1ms/step - accuracy: 0.9015 - loss: 0.0030
    Epoch 54/100
    438/438 - 1s - 1ms/step - accuracy: 0.9026 - loss: 0.0030
    Epoch 55/100
    438/438 - 1s - 1ms/step - accuracy: 0.9030 - loss: 0.0030
    Epoch 56/100
    438/438 - 1s - 1ms/step - accuracy: 0.9012 - loss: 0.0030
    Epoch 57/100
    438/438 - 1s - 1ms/step - accuracy: 0.9004 - loss: 0.0030
    Epoch 58/100
    438/438 - 1s - 1ms/step - accuracy: 0.9003 - loss: 0.0030
    Epoch 59/100
    438/438 - 1s - 1ms/step - accuracy: 0.9025 - loss: 0.0030
    Epoch 60/100
    438/438 - 1s - 1ms/step - accuracy: 0.9014 - loss: 0.0030
    Epoch 61/100
    438/438 - 1s - 1ms/step - accuracy: 0.9019 - loss: 0.0030
    Epoch 62/100
    438/438 - 1s - 1ms/step - accuracy: 0.9035 - loss: 0.0030
    Epoch 63/100
    438/438 - 1s - 1ms/step - accuracy: 0.9044 - loss: 0.0030
    Epoch 64/100
    438/438 - 1s - 1ms/step - accuracy: 0.9037 - loss: 0.0030
    Epoch 65/100
    438/438 - 1s - 1ms/step - accuracy: 0.9045 - loss: 0.0030
    Epoch 66/100
    438/438 - 1s - 1ms/step - accuracy: 0.9046 - loss: 0.0030
    Epoch 67/100
    438/438 - 1s - 1ms/step - accuracy: 0.9037 - loss: 0.0030
    Epoch 68/100
    438/438 - 1s - 1ms/step - accuracy: 0.9046 - loss: 0.0030
    Epoch 69/100
    438/438 - 1s - 1ms/step - accuracy: 0.9048 - loss: 0.0030
    Epoch 70/100
    438/438 - 1s - 1ms/step - accuracy: 0.9051 - loss: 0.0030
    Epoch 71/100
    438/438 - 1s - 1ms/step - accuracy: 0.9047 - loss: 0.0030
    Epoch 72/100
    438/438 - 1s - 1ms/step - accuracy: 0.9052 - loss: 0.0030
    Epoch 73/100
    438/438 - 1s - 1ms/step - accuracy: 0.9041 - loss: 0.0030
    Epoch 74/100
    438/438 - 1s - 1ms/step - accuracy: 0.9039 - loss: 0.0030
    Epoch 75/100
    438/438 - 1s - 1ms/step - accuracy: 0.9036 - loss: 0.0030
    Epoch 76/100
    438/438 - 1s - 1ms/step - accuracy: 0.9045 - loss: 0.0030
    Epoch 77/100
    438/438 - 1s - 1ms/step - accuracy: 0.9042 - loss: 0.0030
    Epoch 78/100
    438/438 - 1s - 1ms/step - accuracy: 0.9042 - loss: 0.0030
    Epoch 79/100
    438/438 - 1s - 1ms/step - accuracy: 0.9039 - loss: 0.0030
    Epoch 80/100
    438/438 - 1s - 1ms/step - accuracy: 0.9052 - loss: 0.0030
    Epoch 81/100
    438/438 - 1s - 1ms/step - accuracy: 0.9033 - loss: 0.0030
    Epoch 82/100
    438/438 - 1s - 1ms/step - accuracy: 0.9042 - loss: 0.0030
    Epoch 83/100
    438/438 - 1s - 1ms/step - accuracy: 0.9035 - loss: 0.0030
    Epoch 84/100
    438/438 - 1s - 1ms/step - accuracy: 0.9039 - loss: 0.0030
    Epoch 85/100
    438/438 - 1s - 1ms/step - accuracy: 0.9049 - loss: 0.0030
    Epoch 86/100
    438/438 - 1s - 1ms/step - accuracy: 0.9032 - loss: 0.0030
    Epoch 87/100
    438/438 - 1s - 1ms/step - accuracy: 0.9048 - loss: 0.0030
    Epoch 88/100
    438/438 - 1s - 1ms/step - accuracy: 0.9041 - loss: 0.0030
    Epoch 89/100
    438/438 - 1s - 1ms/step - accuracy: 0.9042 - loss: 0.0030
    Epoch 90/100
    438/438 - 1s - 1ms/step - accuracy: 0.9041 - loss: 0.0030
    Epoch 91/100
    438/438 - 1s - 1ms/step - accuracy: 0.9054 - loss: 0.0030
    Epoch 92/100
    438/438 - 1s - 1ms/step - accuracy: 0.9037 - loss: 0.0030
    Epoch 93/100
    438/438 - 1s - 1ms/step - accuracy: 0.9043 - loss: 0.0030
    Epoch 94/100
    438/438 - 1s - 1ms/step - accuracy: 0.9035 - loss: 0.0030
    Epoch 95/100
    438/438 - 1s - 1ms/step - accuracy: 0.9033 - loss: 0.0030
    Epoch 96/100
    438/438 - 1s - 1ms/step - accuracy: 0.9032 - loss: 0.0030
    Epoch 97/100
    438/438 - 1s - 1ms/step - accuracy: 0.9040 - loss: 0.0030
    Epoch 98/100
    438/438 - 1s - 1ms/step - accuracy: 0.9035 - loss: 0.0030
    Epoch 99/100
    438/438 - 1s - 1ms/step - accuracy: 0.9037 - loss: 0.0030
    Epoch 100/100
    438/438 - 1s - 1ms/step - accuracy: 0.9047 - loss: 0.0030
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.01401778 0.03254340 0.07920297 0.02653983 

![](CNN_SIR_all_files/figure-commonmark/give%20parameters%20and%20run-1.png)

![](CNN_SIR_all_files/figure-commonmark/give%20parameters%20and%20run-2.png)

# Section 2

Now we can Run a CNN model to find the best parameters we can use for
our CNN model to perform

``` r
build_cnn_model <- function(input_shape, output_units, filters, kernel_size, activation, dense_units) {
  model <- keras3::keras_model_sequential() %>%
    keras3::layer_conv_2d(
      filters = filters,                      # Use dynamic filters
      kernel_size = kernel_size,              # Use dynamic kernel size
      activation = activation,                # Use dynamic activation
      input_shape = c(input_shape, 1)         # Assuming single-channel input (e.g., grayscale)
    ) %>%
    keras3::layer_max_pooling_2d(pool_size = 2, padding = 'same') %>%
    keras3::layer_flatten() %>%
    keras3::layer_dense(units = dense_units, activation = activation) %>%
    keras3::layer_dense(units = output_units, activation = 'sigmoid')  # Output layer

  model %>% compile(
    optimizer = 'adam', 
    loss = 'mse', 
    metrics = 'accuracy'
  )

  return(model)
}
```

``` r
#|label: main_pipeline for tuning
main_pipeline <- function(N, n, ndays, ncores) {
  
  # Generate theta and seeds
  theta <- generate_theta(N, n)
  seeds <- sample.int(.Machine$integer.max, N, TRUE)
  
  # Run simulations
  matrices <- run_simulations(N, n, ndays, ncores, theta, seeds)
  
  # Filter non-null elements
  filtered_data <- filter_non_null(matrices, theta)
  matrices <- filtered_data$matrices
  theta <- filtered_data$theta
  N <- filtered_data$N
  
  # Prepare data for TensorFlow
  arrays_1d <- prepare_data_for_tensorflow(matrices, N)
  
  # Save theta and simulations data
  theta2 <- copy(theta)
  theta2[, crate := plogis(crate / 10)]  # Apply logit transform to crate
  saveRDS(list(theta = theta2, simulations = arrays_1d), file = "calibration/sir.rds", compress = TRUE)
  
  # Split data into training and testing sets
  data_split <- split_data(arrays_1d, theta2, N)
  train <- data_split$train
  test <- data_split$test
  
  # Return train and test sets for future use
  return(list(train = train, test = test, theta2 = theta2, arrays_1d = arrays_1d))
}
```

The `main_pipeline` function streamlines the process of generating and
preparing data for machine learning models. By returning the prepared
datasets, it allows for modularity and flexibility, enabling you to
focus on building and evaluating models without worrying about data
preparation each time.

# Function to experiment with different CNN configurations and track performance

The `experiment_cnn_models` function systematically experiments with
different configurations (hyperparameters) of a Convolutional Neural
Network (CNN) to determine which combination yields the best
performance, measured by the Mean Absolute Error (MAE) on the test
dataset.

``` r
experiment_cnn_models <- function(train_data, test_data, theta, input_shape, output_units, epochs = 10) {
  # Define the grid of hyperparameters to explore
  filter_sizes <- c(16, 32, 64)         # Number of filters
  kernel_sizes <- list(c(3, 3), c(5, 5), c(3,5)) # Kernel sizes
  activations <- c("relu", "linear")     # Activation functions
  dense_units <- c(32, 64, 128)          # Dense layer units
  
  # Store the results in a data frame for comparison
  results <- data.table(
    filters = integer(),
    kernel_size = character(),
    activation = character(),
    dense_units = integer(),
    MAE = numeric()
  )
  
  # Loop over all combinations of hyperparameters
  for (filters in filter_sizes) {
    for (kernel_size in kernel_sizes) {
      for (activation in activations) {
        for (dense in dense_units) {
          cat(sprintf("\nTraining model with filters=%d, kernel_size=%s, activation=%s, dense_units=%d\n",
                      filters, paste(kernel_size, collapse="x"), activation, dense))
          
          # Build the model with current hyperparameters
          model <- build_cnn_model(
            input_shape = input_shape, 
            output_units = output_units,
            filters = filters, 
            kernel_size = kernel_size, 
            activation = activation, 
            dense_units = dense
          )
          
          # Train the model
          train_model(model, train_data, epochs = epochs)
          
          # Evaluate the model
          eval_results <- evaluate_model(model, test_data, theta)
          pred <- eval_results$pred
          MAEs <- eval_results$MAEs
          
          # Store the configuration and the MAE result
          results <- rbind(results, data.table(
            filters = filters,
            kernel_size = paste(kernel_size, collapse = "x"),
            activation = activation,
            dense_units = dense,
            MAE = mean(MAEs)  # Storing the mean MAE for comparison
          ))
          
          # Output the performance
          cat("Mean MAE for this configuration:", mean(MAEs), "\n")
        }
      }
    }
  }
  
  # Return the results data table with all configurations and their performance
  return(results)
}
```

**Function Breakdown**:

1.  **Hyperparameter Grid Definition**:

    - **`filter_sizes`**: Specifies different numbers of filters to try
      (16, 32, 64).

    - **`kernel_sizes`**: Lists different kernel size combinations to
      test (3x3, 5x5, 3x5).

    - **`activations`**: Includes activation functions to experiment
      with (“relu”, “linear”).

    - **`dense_units`**: Defines different sizes for the dense layer
      (32, 64, 128).

2.  **Results Initialization**:

    - Creates an empty `data.table` called `results` to store the
      performance metrics of each model configuration.

3.  **Hyperparameter Combination Loop**:

    - Nested loops iterate over every combination of the
      hyperparameters.

    - For each combination:

      - **Model Training**:

        - Prints the current configuration being trained.

        - Builds the CNN model using the `build_cnn_model` function with
          the current hyperparameters.

        - Trains the model on the provided training data using
          `train_model`.

      - **Model Evaluation**:

        - Evaluates the model on the test data using `evaluate_model`.

        - Calculates the mean MAE from the evaluation results.

      - **Results Recording**:

        - Appends the current hyperparameter settings and the
          corresponding mean MAE to the `results` table.

        - Prints the mean MAE for the current configuration.

4.  **Return Statement**:

    - After all combinations have been evaluated, the function returns
      the `results` table containing all configurations and their mean
      MAE scores.

# Running the Process to Find the Best Model

``` r
N <- 2e4
n <- 5000
ndays <- 50
ncores <- 20

# Prepare the dataset
pipeline_data <- main_pipeline(N, n, ndays, ncores)
train <- pipeline_data$train
test <- pipeline_data$test
theta2 <- pipeline_data$theta2
arrays_1d <- pipeline_data$arrays_1d

# Define the input shape and number of output units
input_shape <- dim(arrays_1d)[-1]  # Shape excluding the batch size
output_units <- ncol(theta2)

# Run the CNN model experiments
results <- experiment_cnn_models(train, test, theta2, input_shape, output_units)
```


    Training model with filters=16, kernel_size=3x3, activation=relu, dense_units=32
    Epoch 1/10
    438/438 - 1s - 3ms/step - accuracy: 0.7484 - loss: 0.0233
    Epoch 2/10
    438/438 - 1s - 1ms/step - accuracy: 0.8330 - loss: 0.0058
    Epoch 3/10
    438/438 - 1s - 1ms/step - accuracy: 0.8515 - loss: 0.0043
    Epoch 4/10
    438/438 - 1s - 1ms/step - accuracy: 0.8641 - loss: 0.0038
    Epoch 5/10
    438/438 - 1s - 1ms/step - accuracy: 0.8708 - loss: 0.0036
    Epoch 6/10
    438/438 - 1s - 1ms/step - accuracy: 0.8765 - loss: 0.0034
    Epoch 7/10
    438/438 - 1s - 1ms/step - accuracy: 0.8809 - loss: 0.0033
    Epoch 8/10
    438/438 - 1s - 1ms/step - accuracy: 0.8885 - loss: 0.0032
    Epoch 9/10
    438/438 - 1s - 1ms/step - accuracy: 0.8939 - loss: 0.0031
    Epoch 10/10
    438/438 - 1s - 1ms/step - accuracy: 0.8970 - loss: 0.0031
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.02147163 0.03139945 0.07254959 0.02469232 
    Mean MAE for this configuration: 0.03752825 

    Training model with filters=16, kernel_size=3x3, activation=relu, dense_units=64
    Epoch 1/10
    438/438 - 1s - 3ms/step - accuracy: 0.7342 - loss: 0.0419
    Epoch 2/10
    438/438 - 1s - 1ms/step - accuracy: 0.8134 - loss: 0.0076
    Epoch 3/10
    438/438 - 1s - 1ms/step - accuracy: 0.8464 - loss: 0.0050
    Epoch 4/10
    438/438 - 1s - 1ms/step - accuracy: 0.8637 - loss: 0.0043
    Epoch 5/10
    438/438 - 1s - 1ms/step - accuracy: 0.8727 - loss: 0.0038
    Epoch 6/10
    438/438 - 1s - 1ms/step - accuracy: 0.8798 - loss: 0.0035
    Epoch 7/10
    438/438 - 1s - 1ms/step - accuracy: 0.8879 - loss: 0.0033
    Epoch 8/10
    438/438 - 1s - 1ms/step - accuracy: 0.8897 - loss: 0.0032
    Epoch 9/10
    438/438 - 1s - 1ms/step - accuracy: 0.8920 - loss: 0.0031
    Epoch 10/10
    438/438 - 1s - 1ms/step - accuracy: 0.8946 - loss: 0.0031
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.01946953 0.03404719 0.07397494 0.03093267 
    Mean MAE for this configuration: 0.03960608 

    Training model with filters=16, kernel_size=3x3, activation=relu, dense_units=128
    Epoch 1/10
    438/438 - 1s - 3ms/step - accuracy: 0.7433 - loss: 0.0671
    Epoch 2/10
    438/438 - 1s - 2ms/step - accuracy: 0.7874 - loss: 0.0174
    Epoch 3/10
    438/438 - 1s - 2ms/step - accuracy: 0.8415 - loss: 0.0053
    Epoch 4/10
    438/438 - 1s - 2ms/step - accuracy: 0.8618 - loss: 0.0044
    Epoch 5/10
    438/438 - 1s - 2ms/step - accuracy: 0.8689 - loss: 0.0040
    Epoch 6/10
    438/438 - 1s - 2ms/step - accuracy: 0.8757 - loss: 0.0037
    Epoch 7/10
    438/438 - 1s - 2ms/step - accuracy: 0.8837 - loss: 0.0034
    Epoch 8/10
    438/438 - 1s - 2ms/step - accuracy: 0.8901 - loss: 0.0033
    Epoch 9/10
    438/438 - 1s - 2ms/step - accuracy: 0.8946 - loss: 0.0032
    Epoch 10/10
    438/438 - 1s - 2ms/step - accuracy: 0.8963 - loss: 0.0031
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.01529082 0.03229686 0.07576668 0.02316660 
    Mean MAE for this configuration: 0.03663024 

    Training model with filters=16, kernel_size=3x3, activation=linear, dense_units=32
    Epoch 1/10
    438/438 - 2s - 4ms/step - accuracy: 0.8155 - loss: 0.1343
    Epoch 2/10
    438/438 - 1s - 1ms/step - accuracy: 0.7544 - loss: 0.0299
    Epoch 3/10
    438/438 - 1s - 1ms/step - accuracy: 0.8491 - loss: 0.0054
    Epoch 4/10
    438/438 - 1s - 1ms/step - accuracy: 0.8634 - loss: 0.0045
    Epoch 5/10
    438/438 - 1s - 1ms/step - accuracy: 0.8702 - loss: 0.0042
    Epoch 6/10
    438/438 - 1s - 1ms/step - accuracy: 0.8737 - loss: 0.0039
    Epoch 7/10
    438/438 - 1s - 1ms/step - accuracy: 0.8777 - loss: 0.0038
    Epoch 8/10
    438/438 - 1s - 1ms/step - accuracy: 0.8790 - loss: 0.0037
    Epoch 9/10
    438/438 - 1s - 1ms/step - accuracy: 0.8804 - loss: 0.0036
    Epoch 10/10
    438/438 - 1s - 1ms/step - accuracy: 0.8803 - loss: 0.0035
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.02447173 0.03291609 0.07738921 0.03115186 
    Mean MAE for this configuration: 0.04148222 

    Training model with filters=16, kernel_size=3x3, activation=linear, dense_units=64
    Epoch 1/10
    438/438 - 1s - 3ms/step - accuracy: 0.8124 - loss: 0.1313
    Epoch 2/10
    438/438 - 1s - 1ms/step - accuracy: 0.7933 - loss: 0.1232
    Epoch 3/10
    438/438 - 1s - 1ms/step - accuracy: 0.7306 - loss: 0.0449
    Epoch 4/10
    438/438 - 1s - 1ms/step - accuracy: 0.8451 - loss: 0.0058
    Epoch 5/10
    438/438 - 1s - 1ms/step - accuracy: 0.8661 - loss: 0.0044
    Epoch 6/10
    438/438 - 1s - 1ms/step - accuracy: 0.8719 - loss: 0.0041
    Epoch 7/10
    438/438 - 1s - 1ms/step - accuracy: 0.8756 - loss: 0.0039
    Epoch 8/10
    438/438 - 1s - 1ms/step - accuracy: 0.8786 - loss: 0.0038
    Epoch 9/10
    438/438 - 1s - 1ms/step - accuracy: 0.8782 - loss: 0.0037
    Epoch 10/10
    438/438 - 1s - 1ms/step - accuracy: 0.8801 - loss: 0.0037
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.02517450 0.03631274 0.07748089 0.04171583 
    Mean MAE for this configuration: 0.04517099 

    Training model with filters=16, kernel_size=3x3, activation=linear, dense_units=128
    Epoch 1/10
    438/438 - 1s - 3ms/step - accuracy: 0.8127 - loss: 0.1404
    Epoch 2/10
    438/438 - 1s - 1ms/step - accuracy: 0.8267 - loss: 0.1337
    Epoch 3/10
    438/438 - 1s - 1ms/step - accuracy: 0.8202 - loss: 0.1434
    Epoch 4/10
    438/438 - 1s - 1ms/step - accuracy: 0.8216 - loss: 0.1349
    Epoch 5/10
    438/438 - 1s - 1ms/step - accuracy: 0.8208 - loss: 0.1327
    Epoch 6/10
    438/438 - 1s - 1ms/step - accuracy: 0.8196 - loss: 0.1315
    Epoch 7/10
    438/438 - 1s - 1ms/step - accuracy: 0.8251 - loss: 0.1303
    Epoch 8/10
    438/438 - 1s - 1ms/step - accuracy: 0.8286 - loss: 0.1299
    Epoch 9/10
    438/438 - 1s - 1ms/step - accuracy: 0.8056 - loss: 0.1230
    Epoch 10/10
    438/438 - 1s - 1ms/step - accuracy: 0.7201 - loss: 0.0693
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.20936182 0.03776354 0.29914916 0.06663654 
    Mean MAE for this configuration: 0.1532278 

    Training model with filters=16, kernel_size=5x5, activation=relu, dense_units=32
    Epoch 1/10
    438/438 - 1s - 3ms/step - accuracy: 0.7740 - loss: 0.0319
    Epoch 2/10
    438/438 - 1s - 1ms/step - accuracy: 0.8432 - loss: 0.0057
    Epoch 3/10
    438/438 - 1s - 1ms/step - accuracy: 0.8639 - loss: 0.0042
    Epoch 4/10
    438/438 - 1s - 1ms/step - accuracy: 0.8789 - loss: 0.0036
    Epoch 5/10
    438/438 - 1s - 1ms/step - accuracy: 0.8876 - loss: 0.0034
    Epoch 6/10
    438/438 - 1s - 1ms/step - accuracy: 0.8936 - loss: 0.0033
    Epoch 7/10
    438/438 - 1s - 1ms/step - accuracy: 0.8941 - loss: 0.0032
    Epoch 8/10
    438/438 - 1s - 1ms/step - accuracy: 0.8949 - loss: 0.0031
    Epoch 9/10
    438/438 - 1s - 1ms/step - accuracy: 0.8956 - loss: 0.0031
    Epoch 10/10
    438/438 - 1s - 1ms/step - accuracy: 0.8969 - loss: 0.0031
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.01399086 0.03409769 0.07380494 0.02533926 
    Mean MAE for this configuration: 0.03680819 

    Training model with filters=16, kernel_size=5x5, activation=relu, dense_units=64
    Epoch 1/10
    438/438 - 1s - 3ms/step - accuracy: 0.7507 - loss: 0.0401
    Epoch 2/10
    438/438 - 1s - 1ms/step - accuracy: 0.8217 - loss: 0.0057
    Epoch 3/10
    438/438 - 1s - 1ms/step - accuracy: 0.8539 - loss: 0.0041
    Epoch 4/10
    438/438 - 1s - 1ms/step - accuracy: 0.8688 - loss: 0.0036
    Epoch 5/10
    438/438 - 1s - 1ms/step - accuracy: 0.8798 - loss: 0.0034
    Epoch 6/10
    438/438 - 1s - 1ms/step - accuracy: 0.8870 - loss: 0.0032
    Epoch 7/10
    438/438 - 1s - 1ms/step - accuracy: 0.8921 - loss: 0.0031
    Epoch 8/10
    438/438 - 1s - 1ms/step - accuracy: 0.8997 - loss: 0.0031
    Epoch 9/10
    438/438 - 1s - 1ms/step - accuracy: 0.9028 - loss: 0.0030
    Epoch 10/10
    438/438 - 1s - 1ms/step - accuracy: 0.9052 - loss: 0.0029
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.01402642 0.03514259 0.07508233 0.01858595 
    Mean MAE for this configuration: 0.03570932 

    Training model with filters=16, kernel_size=5x5, activation=relu, dense_units=128
    Epoch 1/10
    438/438 - 1s - 3ms/step - accuracy: 0.7392 - loss: 0.0482
    Epoch 2/10
    438/438 - 1s - 1ms/step - accuracy: 0.8258 - loss: 0.0141
    Epoch 3/10
    438/438 - 1s - 1ms/step - accuracy: 0.8641 - loss: 0.0042
    Epoch 4/10
    438/438 - 1s - 1ms/step - accuracy: 0.8778 - loss: 0.0036
    Epoch 5/10
    438/438 - 1s - 1ms/step - accuracy: 0.8882 - loss: 0.0033
    Epoch 6/10
    438/438 - 1s - 1ms/step - accuracy: 0.8927 - loss: 0.0032
    Epoch 7/10
    438/438 - 1s - 1ms/step - accuracy: 0.9003 - loss: 0.0030
    Epoch 8/10
    438/438 - 1s - 1ms/step - accuracy: 0.9020 - loss: 0.0030
    Epoch 9/10
    438/438 - 1s - 1ms/step - accuracy: 0.9033 - loss: 0.0029
    Epoch 10/10
    438/438 - 1s - 1ms/step - accuracy: 0.9040 - loss: 0.0029
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.01412662 0.03296492 0.07363224 0.02382164 
    Mean MAE for this configuration: 0.03613636 

    Training model with filters=16, kernel_size=5x5, activation=linear, dense_units=32
    Epoch 1/10
    438/438 - 1s - 3ms/step - accuracy: 0.7344 - loss: 0.0621
    Epoch 2/10
    438/438 - 1s - 1ms/step - accuracy: 0.8143 - loss: 0.0127
    Epoch 3/10
    438/438 - 1s - 1ms/step - accuracy: 0.8652 - loss: 0.0046
    Epoch 4/10
    438/438 - 1s - 1ms/step - accuracy: 0.8691 - loss: 0.0040
    Epoch 5/10
    438/438 - 1s - 1ms/step - accuracy: 0.8728 - loss: 0.0039
    Epoch 6/10
    438/438 - 1s - 1ms/step - accuracy: 0.8744 - loss: 0.0038
    Epoch 7/10
    438/438 - 1s - 1ms/step - accuracy: 0.8779 - loss: 0.0038
    Epoch 8/10
    438/438 - 1s - 1ms/step - accuracy: 0.8800 - loss: 0.0037
    Epoch 9/10
    438/438 - 1s - 1ms/step - accuracy: 0.8824 - loss: 0.0036
    Epoch 10/10
    438/438 - 1s - 1ms/step - accuracy: 0.8849 - loss: 0.0036
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.02714988 0.03230574 0.07822272 0.03135192 
    Mean MAE for this configuration: 0.04225757 

    Training model with filters=16, kernel_size=5x5, activation=linear, dense_units=64
    Epoch 1/10
    438/438 - 2s - 4ms/step - accuracy: 0.7830 - loss: 0.0902
    Epoch 2/10
    438/438 - 1s - 1ms/step - accuracy: 0.8248 - loss: 0.0116
    Epoch 3/10
    438/438 - 1s - 1ms/step - accuracy: 0.8659 - loss: 0.0047
    Epoch 4/10
    438/438 - 1s - 1ms/step - accuracy: 0.8713 - loss: 0.0042
    Epoch 5/10
    438/438 - 1s - 1ms/step - accuracy: 0.8758 - loss: 0.0040
    Epoch 6/10
    438/438 - 1s - 1ms/step - accuracy: 0.8779 - loss: 0.0039
    Epoch 7/10
    438/438 - 1s - 1ms/step - accuracy: 0.8772 - loss: 0.0039
    Epoch 8/10
    438/438 - 1s - 1ms/step - accuracy: 0.8763 - loss: 0.0038
    Epoch 9/10
    438/438 - 1s - 1ms/step - accuracy: 0.8784 - loss: 0.0037
    Epoch 10/10
    438/438 - 1s - 1ms/step - accuracy: 0.8797 - loss: 0.0036
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.03438376 0.03374374 0.07700405 0.03190838 
    Mean MAE for this configuration: 0.04425998 

    Training model with filters=16, kernel_size=5x5, activation=linear, dense_units=128
    Epoch 1/10
    438/438 - 1s - 3ms/step - accuracy: 0.7875 - loss: 0.1148
    Epoch 2/10
    438/438 - 1s - 1ms/step - accuracy: 0.7932 - loss: 0.0243
    Epoch 3/10
    438/438 - 1s - 1ms/step - accuracy: 0.8587 - loss: 0.0050
    Epoch 4/10
    438/438 - 1s - 1ms/step - accuracy: 0.8690 - loss: 0.0044
    Epoch 5/10
    438/438 - 1s - 1ms/step - accuracy: 0.8724 - loss: 0.0041
    Epoch 6/10
    438/438 - 1s - 1ms/step - accuracy: 0.8714 - loss: 0.0039
    Epoch 7/10
    438/438 - 1s - 1ms/step - accuracy: 0.8741 - loss: 0.0038
    Epoch 8/10
    438/438 - 1s - 1ms/step - accuracy: 0.8767 - loss: 0.0037
    Epoch 9/10
    438/438 - 1s - 1ms/step - accuracy: 0.8752 - loss: 0.0037
    Epoch 10/10
    438/438 - 1s - 1ms/step - accuracy: 0.8764 - loss: 0.0036
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.01753437 0.03516706 0.07625611 0.03154935 
    Mean MAE for this configuration: 0.04012672 

    Training model with filters=16, kernel_size=3x5, activation=relu, dense_units=32
    Epoch 1/10
    438/438 - 1s - 3ms/step - accuracy: 0.7486 - loss: 0.0281
    Epoch 2/10
    438/438 - 1s - 1ms/step - accuracy: 0.8216 - loss: 0.0052
    Epoch 3/10
    438/438 - 1s - 1ms/step - accuracy: 0.8556 - loss: 0.0041
    Epoch 4/10
    438/438 - 1s - 1ms/step - accuracy: 0.8706 - loss: 0.0037
    Epoch 5/10
    438/438 - 1s - 1ms/step - accuracy: 0.8791 - loss: 0.0034
    Epoch 6/10
    438/438 - 1s - 1ms/step - accuracy: 0.8862 - loss: 0.0033
    Epoch 7/10
    438/438 - 1s - 1ms/step - accuracy: 0.8907 - loss: 0.0032
    Epoch 8/10
    438/438 - 1s - 1ms/step - accuracy: 0.8924 - loss: 0.0031
    Epoch 9/10
    438/438 - 1s - 1ms/step - accuracy: 0.8948 - loss: 0.0031
    Epoch 10/10
    438/438 - 1s - 1ms/step - accuracy: 0.8949 - loss: 0.0030
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.01574871 0.03364821 0.07363045 0.03009332 
    Mean MAE for this configuration: 0.03828017 

    Training model with filters=16, kernel_size=3x5, activation=relu, dense_units=64
    Epoch 1/10
    438/438 - 1s - 3ms/step - accuracy: 0.7675 - loss: 0.0321
    Epoch 2/10
    438/438 - 1s - 1ms/step - accuracy: 0.8414 - loss: 0.0050
    Epoch 3/10
    438/438 - 1s - 1ms/step - accuracy: 0.8656 - loss: 0.0040
    Epoch 4/10
    438/438 - 1s - 1ms/step - accuracy: 0.8794 - loss: 0.0036
    Epoch 5/10
    438/438 - 1s - 1ms/step - accuracy: 0.8864 - loss: 0.0034
    Epoch 6/10
    438/438 - 1s - 1ms/step - accuracy: 0.8913 - loss: 0.0032
    Epoch 7/10
    438/438 - 1s - 1ms/step - accuracy: 0.8947 - loss: 0.0031
    Epoch 8/10
    438/438 - 1s - 1ms/step - accuracy: 0.8964 - loss: 0.0030
    Epoch 9/10
    438/438 - 1s - 1ms/step - accuracy: 0.9003 - loss: 0.0030
    Epoch 10/10
    438/438 - 1s - 1ms/step - accuracy: 0.9025 - loss: 0.0029
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.01875649 0.03524869 0.07490055 0.02041817 
    Mean MAE for this configuration: 0.03733097 

    Training model with filters=16, kernel_size=3x5, activation=relu, dense_units=128
    Epoch 1/10
    438/438 - 1s - 3ms/step - accuracy: 0.7447 - loss: 0.0501
    Epoch 2/10
    438/438 - 1s - 2ms/step - accuracy: 0.8268 - loss: 0.0192
    Epoch 3/10
    438/438 - 1s - 2ms/step - accuracy: 0.8516 - loss: 0.0098
    Epoch 4/10
    438/438 - 1s - 2ms/step - accuracy: 0.8793 - loss: 0.0037
    Epoch 5/10
    438/438 - 1s - 2ms/step - accuracy: 0.8896 - loss: 0.0033
    Epoch 6/10
    438/438 - 1s - 2ms/step - accuracy: 0.8956 - loss: 0.0032
    Epoch 7/10
    438/438 - 1s - 2ms/step - accuracy: 0.8992 - loss: 0.0031
    Epoch 8/10
    438/438 - 1s - 2ms/step - accuracy: 0.9005 - loss: 0.0030
    Epoch 9/10
    438/438 - 1s - 2ms/step - accuracy: 0.9030 - loss: 0.0029
    Epoch 10/10
    438/438 - 1s - 2ms/step - accuracy: 0.9055 - loss: 0.0029
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.01629789 0.03287565 0.07494970 0.02011277 
    Mean MAE for this configuration: 0.036059 

    Training model with filters=16, kernel_size=3x5, activation=linear, dense_units=32
    Epoch 1/10
    438/438 - 1s - 3ms/step - accuracy: 0.7271 - loss: 0.0617
    Epoch 2/10
    438/438 - 1s - 1ms/step - accuracy: 0.8189 - loss: 0.0151
    Epoch 3/10
    438/438 - 1s - 1ms/step - accuracy: 0.8622 - loss: 0.0047
    Epoch 4/10
    438/438 - 1s - 1ms/step - accuracy: 0.8703 - loss: 0.0041
    Epoch 5/10
    438/438 - 1s - 1ms/step - accuracy: 0.8739 - loss: 0.0039
    Epoch 6/10
    438/438 - 1s - 1ms/step - accuracy: 0.8768 - loss: 0.0038
    Epoch 7/10
    438/438 - 1s - 1ms/step - accuracy: 0.8779 - loss: 0.0037
    Epoch 8/10
    438/438 - 1s - 1ms/step - accuracy: 0.8812 - loss: 0.0036
    Epoch 9/10
    438/438 - 1s - 1ms/step - accuracy: 0.8827 - loss: 0.0036
    Epoch 10/10
    438/438 - 1s - 1ms/step - accuracy: 0.8834 - loss: 0.0035
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.03202427 0.03337585 0.07844307 0.03074780 
    Mean MAE for this configuration: 0.04364775 

    Training model with filters=16, kernel_size=3x5, activation=linear, dense_units=64
    Epoch 1/10
    438/438 - 1s - 3ms/step - accuracy: 0.7669 - loss: 0.1003
    Epoch 2/10
    438/438 - 1s - 1ms/step - accuracy: 0.8030 - loss: 0.0282
    Epoch 3/10
    438/438 - 1s - 1ms/step - accuracy: 0.8596 - loss: 0.0053
    Epoch 4/10
    438/438 - 1s - 1ms/step - accuracy: 0.8670 - loss: 0.0044
    Epoch 5/10
    438/438 - 1s - 1ms/step - accuracy: 0.8737 - loss: 0.0041
    Epoch 6/10
    438/438 - 1s - 1ms/step - accuracy: 0.8751 - loss: 0.0040
    Epoch 7/10
    438/438 - 1s - 1ms/step - accuracy: 0.8785 - loss: 0.0038
    Epoch 8/10
    438/438 - 1s - 1ms/step - accuracy: 0.8803 - loss: 0.0037
    Epoch 9/10
    438/438 - 1s - 1ms/step - accuracy: 0.8806 - loss: 0.0037
    Epoch 10/10
    438/438 - 1s - 1ms/step - accuracy: 0.8803 - loss: 0.0036
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.02492167 0.03501169 0.08069073 0.03322711 
    Mean MAE for this configuration: 0.0434628 

    Training model with filters=16, kernel_size=3x5, activation=linear, dense_units=128
    Epoch 1/10
    438/438 - 1s - 3ms/step - accuracy: 0.8167 - loss: 0.1286
    Epoch 2/10
    438/438 - 1s - 1ms/step - accuracy: 0.7216 - loss: 0.0649
    Epoch 3/10
    438/438 - 1s - 1ms/step - accuracy: 0.8442 - loss: 0.0061
    Epoch 4/10
    438/438 - 1s - 1ms/step - accuracy: 0.8661 - loss: 0.0045
    Epoch 5/10
    438/438 - 1s - 1ms/step - accuracy: 0.8739 - loss: 0.0042
    Epoch 6/10
    438/438 - 1s - 1ms/step - accuracy: 0.8746 - loss: 0.0040
    Epoch 7/10
    438/438 - 1s - 1ms/step - accuracy: 0.8763 - loss: 0.0038
    Epoch 8/10
    438/438 - 1s - 1ms/step - accuracy: 0.8804 - loss: 0.0036
    Epoch 9/10
    438/438 - 1s - 1ms/step - accuracy: 0.8800 - loss: 0.0036
    Epoch 10/10
    438/438 - 1s - 1ms/step - accuracy: 0.8809 - loss: 0.0035
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.02512083 0.03251516 0.07563982 0.03460740 
    Mean MAE for this configuration: 0.0419708 

    Training model with filters=32, kernel_size=3x3, activation=relu, dense_units=32
    Epoch 1/10
    438/438 - 2s - 4ms/step - accuracy: 0.7447 - loss: 0.0299
    Epoch 2/10
    438/438 - 1s - 2ms/step - accuracy: 0.8135 - loss: 0.0064
    Epoch 3/10
    438/438 - 1s - 2ms/step - accuracy: 0.8477 - loss: 0.0044
    Epoch 4/10
    438/438 - 1s - 2ms/step - accuracy: 0.8644 - loss: 0.0039
    Epoch 5/10
    438/438 - 1s - 2ms/step - accuracy: 0.8757 - loss: 0.0036
    Epoch 6/10
    438/438 - 1s - 2ms/step - accuracy: 0.8820 - loss: 0.0034
    Epoch 7/10
    438/438 - 1s - 2ms/step - accuracy: 0.8865 - loss: 0.0033
    Epoch 8/10
    438/438 - 1s - 2ms/step - accuracy: 0.8911 - loss: 0.0032
    Epoch 9/10
    438/438 - 1s - 2ms/step - accuracy: 0.8944 - loss: 0.0031
    Epoch 10/10
    438/438 - 1s - 2ms/step - accuracy: 0.8981 - loss: 0.0031
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.01907885 0.03312735 0.07318218 0.02534506 
    Mean MAE for this configuration: 0.03768336 

    Training model with filters=32, kernel_size=3x3, activation=relu, dense_units=64
    Epoch 1/10
    438/438 - 2s - 4ms/step - accuracy: 0.7759 - loss: 0.0300
    Epoch 2/10
    438/438 - 1s - 2ms/step - accuracy: 0.8516 - loss: 0.0050
    Epoch 3/10
    438/438 - 1s - 2ms/step - accuracy: 0.8744 - loss: 0.0039
    Epoch 4/10
    438/438 - 1s - 2ms/step - accuracy: 0.8857 - loss: 0.0035
    Epoch 5/10
    438/438 - 1s - 2ms/step - accuracy: 0.8924 - loss: 0.0033
    Epoch 6/10
    438/438 - 1s - 2ms/step - accuracy: 0.8963 - loss: 0.0032
    Epoch 7/10
    438/438 - 1s - 2ms/step - accuracy: 0.8987 - loss: 0.0031
    Epoch 8/10
    438/438 - 1s - 2ms/step - accuracy: 0.9003 - loss: 0.0030
    Epoch 9/10
    438/438 - 1s - 2ms/step - accuracy: 0.9018 - loss: 0.0030
    Epoch 10/10
    438/438 - 1s - 2ms/step - accuracy: 0.9033 - loss: 0.0030
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.01786832 0.03111221 0.07763035 0.02076183 
    Mean MAE for this configuration: 0.03684318 

    Training model with filters=32, kernel_size=3x3, activation=relu, dense_units=128
    Epoch 1/10
    438/438 - 2s - 4ms/step - accuracy: 0.7272 - loss: 0.0421
    Epoch 2/10
    438/438 - 1s - 2ms/step - accuracy: 0.8104 - loss: 0.0205
    Epoch 3/10
    438/438 - 1s - 2ms/step - accuracy: 0.8426 - loss: 0.0185
    Epoch 4/10
    438/438 - 1s - 2ms/step - accuracy: 0.8676 - loss: 0.0112
    Epoch 5/10
    438/438 - 1s - 2ms/step - accuracy: 0.8775 - loss: 0.0036
    Epoch 6/10
    438/438 - 1s - 2ms/step - accuracy: 0.8853 - loss: 0.0033
    Epoch 7/10
    438/438 - 1s - 2ms/step - accuracy: 0.8937 - loss: 0.0032
    Epoch 8/10
    438/438 - 1s - 2ms/step - accuracy: 0.8953 - loss: 0.0031
    Epoch 9/10
    438/438 - 1s - 2ms/step - accuracy: 0.8963 - loss: 0.0030
    Epoch 10/10
    438/438 - 1s - 2ms/step - accuracy: 0.8997 - loss: 0.0030
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.01399858 0.03392113 0.07666556 0.02645521 
    Mean MAE for this configuration: 0.03776012 

    Training model with filters=32, kernel_size=3x3, activation=linear, dense_units=32
    Epoch 1/10
    438/438 - 1s - 3ms/step - accuracy: 0.8075 - loss: 0.1315
    Epoch 2/10
    438/438 - 1s - 2ms/step - accuracy: 0.8207 - loss: 0.1302
    Epoch 3/10
    438/438 - 1s - 2ms/step - accuracy: 0.8140 - loss: 0.1270
    Epoch 4/10
    438/438 - 1s - 1ms/step - accuracy: 0.8203 - loss: 0.1242
    Epoch 5/10
    438/438 - 1s - 1ms/step - accuracy: 0.8192 - loss: 0.1157
    Epoch 6/10
    438/438 - 1s - 1ms/step - accuracy: 0.7955 - loss: 0.0239
    Epoch 7/10
    438/438 - 1s - 1ms/step - accuracy: 0.8719 - loss: 0.0042
    Epoch 8/10
    438/438 - 1s - 1ms/step - accuracy: 0.8804 - loss: 0.0037
    Epoch 9/10
    438/438 - 1s - 1ms/step - accuracy: 0.8837 - loss: 0.0035
    Epoch 10/10
    438/438 - 1s - 1ms/step - accuracy: 0.8873 - loss: 0.0035
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.02188040 0.03264577 0.07962232 0.03052967 
    Mean MAE for this configuration: 0.04116954 

    Training model with filters=32, kernel_size=3x3, activation=linear, dense_units=64
    Epoch 1/10
    438/438 - 2s - 3ms/step - accuracy: 0.8169 - loss: 0.1350
    Epoch 2/10
    438/438 - 1s - 2ms/step - accuracy: 0.8204 - loss: 0.1331
    Epoch 3/10
    438/438 - 1s - 2ms/step - accuracy: 0.8211 - loss: 0.1323
    Epoch 4/10
    438/438 - 1s - 2ms/step - accuracy: 0.8214 - loss: 0.1301
    Epoch 5/10
    438/438 - 1s - 2ms/step - accuracy: 0.8217 - loss: 0.1278
    Epoch 6/10
    438/438 - 1s - 2ms/step - accuracy: 0.8204 - loss: 0.1265
    Epoch 7/10
    438/438 - 1s - 2ms/step - accuracy: 0.8255 - loss: 0.1250
    Epoch 8/10
    438/438 - 1s - 2ms/step - accuracy: 0.8146 - loss: 0.1262
    Epoch 9/10
    438/438 - 1s - 2ms/step - accuracy: 0.8343 - loss: 0.1187
    Epoch 10/10
    438/438 - 1s - 2ms/step - accuracy: 0.8159 - loss: 0.0799
    188/188 - 0s - 1ms/step
       preval     crate     ptran      prec 
    0.2094207 0.3653406 0.2992066 0.0437759 
    Mean MAE for this configuration: 0.2294359 

    Training model with filters=32, kernel_size=3x3, activation=linear, dense_units=128
    Epoch 1/10
    438/438 - 2s - 4ms/step - accuracy: 0.8187 - loss: 0.1377
    Epoch 2/10
    438/438 - 1s - 2ms/step - accuracy: 0.8203 - loss: 0.1395
    Epoch 3/10
    438/438 - 1s - 2ms/step - accuracy: 0.8198 - loss: 0.1343
    Epoch 4/10
    438/438 - 1s - 2ms/step - accuracy: 0.8200 - loss: 0.1321
    Epoch 5/10
    438/438 - 1s - 2ms/step - accuracy: 0.8191 - loss: 0.1333
    Epoch 6/10
    438/438 - 1s - 2ms/step - accuracy: 0.8201 - loss: 0.1327
    Epoch 7/10
    438/438 - 1s - 2ms/step - accuracy: 0.8203 - loss: 0.1338
    Epoch 8/10
    438/438 - 1s - 2ms/step - accuracy: 0.8204 - loss: 0.1330
    Epoch 9/10
    438/438 - 1s - 2ms/step - accuracy: 0.8202 - loss: 0.1327
    Epoch 10/10
    438/438 - 1s - 2ms/step - accuracy: 0.8198 - loss: 0.1323
    188/188 - 0s - 1ms/step
       preval     crate     ptran      prec 
    0.2095984 0.3790727 0.2992234 0.4651087 
    Mean MAE for this configuration: 0.3382508 

    Training model with filters=32, kernel_size=5x5, activation=relu, dense_units=32
    Epoch 1/10
    438/438 - 1s - 3ms/step - accuracy: 0.7682 - loss: 0.0212
    Epoch 2/10
    438/438 - 1s - 1ms/step - accuracy: 0.8313 - loss: 0.0052
    Epoch 3/10
    438/438 - 1s - 1ms/step - accuracy: 0.8540 - loss: 0.0041
    Epoch 4/10
    438/438 - 1s - 1ms/step - accuracy: 0.8657 - loss: 0.0037
    Epoch 5/10
    438/438 - 1s - 1ms/step - accuracy: 0.8744 - loss: 0.0034
    Epoch 6/10
    438/438 - 1s - 1ms/step - accuracy: 0.8842 - loss: 0.0032
    Epoch 7/10
    438/438 - 1s - 1ms/step - accuracy: 0.8930 - loss: 0.0031
    Epoch 8/10
    438/438 - 1s - 1ms/step - accuracy: 0.9005 - loss: 0.0030
    Epoch 9/10
    438/438 - 1s - 1ms/step - accuracy: 0.9029 - loss: 0.0029
    Epoch 10/10
    438/438 - 1s - 1ms/step - accuracy: 0.9045 - loss: 0.0029
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.01694479 0.03180783 0.07457388 0.01833089 
    Mean MAE for this configuration: 0.03541435 

    Training model with filters=32, kernel_size=5x5, activation=relu, dense_units=64
    Epoch 1/10
    438/438 - 1s - 3ms/step - accuracy: 0.7876 - loss: 0.0286
    Epoch 2/10
    438/438 - 1s - 2ms/step - accuracy: 0.8561 - loss: 0.0046
    Epoch 3/10
    438/438 - 1s - 2ms/step - accuracy: 0.8760 - loss: 0.0037
    Epoch 4/10
    438/438 - 1s - 2ms/step - accuracy: 0.8912 - loss: 0.0034
    Epoch 5/10
    438/438 - 1s - 2ms/step - accuracy: 0.8959 - loss: 0.0032
    Epoch 6/10
    438/438 - 1s - 2ms/step - accuracy: 0.9004 - loss: 0.0031
    Epoch 7/10
    438/438 - 1s - 2ms/step - accuracy: 0.9030 - loss: 0.0030
    Epoch 8/10
    438/438 - 1s - 2ms/step - accuracy: 0.9042 - loss: 0.0029
    Epoch 9/10
    438/438 - 1s - 2ms/step - accuracy: 0.9074 - loss: 0.0029
    Epoch 10/10
    438/438 - 1s - 2ms/step - accuracy: 0.9070 - loss: 0.0029
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.01512548 0.03283650 0.07760507 0.02195139 
    Mean MAE for this configuration: 0.03687961 

    Training model with filters=32, kernel_size=5x5, activation=relu, dense_units=128
    Epoch 1/10
    438/438 - 2s - 3ms/step - accuracy: 0.7454 - loss: 0.0402
    Epoch 2/10
    438/438 - 1s - 2ms/step - accuracy: 0.8395 - loss: 0.0090
    Epoch 3/10
    438/438 - 1s - 2ms/step - accuracy: 0.8786 - loss: 0.0038
    Epoch 4/10
    438/438 - 1s - 2ms/step - accuracy: 0.8903 - loss: 0.0034
    Epoch 5/10
    438/438 - 1s - 2ms/step - accuracy: 0.8980 - loss: 0.0032
    Epoch 6/10
    438/438 - 1s - 2ms/step - accuracy: 0.8987 - loss: 0.0031
    Epoch 7/10
    438/438 - 1s - 2ms/step - accuracy: 0.9016 - loss: 0.0030
    Epoch 8/10
    438/438 - 1s - 2ms/step - accuracy: 0.9037 - loss: 0.0030
    Epoch 9/10
    438/438 - 1s - 2ms/step - accuracy: 0.9038 - loss: 0.0029
    Epoch 10/10
    438/438 - 1s - 2ms/step - accuracy: 0.9054 - loss: 0.0029
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.01592746 0.03265400 0.07516760 0.01983388 
    Mean MAE for this configuration: 0.03589573 

    Training model with filters=32, kernel_size=5x5, activation=linear, dense_units=32
    Epoch 1/10
    438/438 - 2s - 4ms/step - accuracy: 0.7401 - loss: 0.0696
    Epoch 2/10
    438/438 - 1s - 1ms/step - accuracy: 0.8496 - loss: 0.0060
    Epoch 3/10
    438/438 - 1s - 1ms/step - accuracy: 0.8720 - loss: 0.0042
    Epoch 4/10
    438/438 - 1s - 1ms/step - accuracy: 0.8772 - loss: 0.0039
    Epoch 5/10
    438/438 - 1s - 1ms/step - accuracy: 0.8803 - loss: 0.0037
    Epoch 6/10
    438/438 - 1s - 1ms/step - accuracy: 0.8808 - loss: 0.0036
    Epoch 7/10
    438/438 - 1s - 1ms/step - accuracy: 0.8839 - loss: 0.0036
    Epoch 8/10
    438/438 - 1s - 1ms/step - accuracy: 0.8834 - loss: 0.0035
    Epoch 9/10
    438/438 - 1s - 1ms/step - accuracy: 0.8848 - loss: 0.0035
    Epoch 10/10
    438/438 - 1s - 1ms/step - accuracy: 0.8844 - loss: 0.0034
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.02960526 0.03389536 0.07744181 0.02845635 
    Mean MAE for this configuration: 0.04234969 

    Training model with filters=32, kernel_size=5x5, activation=linear, dense_units=64
    Epoch 1/10
    438/438 - 1s - 3ms/step - accuracy: 0.7960 - loss: 0.1134
    Epoch 2/10
    438/438 - 1s - 2ms/step - accuracy: 0.8190 - loss: 0.0089
    Epoch 3/10
    438/438 - 1s - 1ms/step - accuracy: 0.8661 - loss: 0.0044
    Epoch 4/10
    438/438 - 1s - 1ms/step - accuracy: 0.8749 - loss: 0.0040
    Epoch 5/10
    438/438 - 1s - 1ms/step - accuracy: 0.8789 - loss: 0.0038
    Epoch 6/10
    438/438 - 1s - 1ms/step - accuracy: 0.8812 - loss: 0.0037
    Epoch 7/10
    438/438 - 1s - 1ms/step - accuracy: 0.8817 - loss: 0.0036
    Epoch 8/10
    438/438 - 1s - 1ms/step - accuracy: 0.8832 - loss: 0.0035
    Epoch 9/10
    438/438 - 1s - 1ms/step - accuracy: 0.8857 - loss: 0.0035
    Epoch 10/10
    438/438 - 1s - 2ms/step - accuracy: 0.8857 - loss: 0.0034
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.01760595 0.03315756 0.07766378 0.03195353 
    Mean MAE for this configuration: 0.0400952 

    Training model with filters=32, kernel_size=5x5, activation=linear, dense_units=128
    Epoch 1/10
    438/438 - 2s - 3ms/step - accuracy: 0.8186 - loss: 0.1390
    Epoch 2/10
    438/438 - 1s - 2ms/step - accuracy: 0.8300 - loss: 0.1347
    Epoch 3/10
    438/438 - 1s - 2ms/step - accuracy: 0.7755 - loss: 0.1231
    Epoch 4/10
    438/438 - 1s - 2ms/step - accuracy: 0.7294 - loss: 0.0437
    Epoch 5/10
    438/438 - 1s - 2ms/step - accuracy: 0.8750 - loss: 0.0045
    Epoch 6/10
    438/438 - 1s - 2ms/step - accuracy: 0.8852 - loss: 0.0039
    Epoch 7/10
    438/438 - 1s - 2ms/step - accuracy: 0.8860 - loss: 0.0037
    Epoch 8/10
    438/438 - 1s - 2ms/step - accuracy: 0.8869 - loss: 0.0036
    Epoch 9/10
    438/438 - 1s - 2ms/step - accuracy: 0.8870 - loss: 0.0035
    Epoch 10/10
    438/438 - 1s - 2ms/step - accuracy: 0.8855 - loss: 0.0034
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.02154632 0.03478256 0.07556873 0.04248825 
    Mean MAE for this configuration: 0.04359647 

    Training model with filters=32, kernel_size=3x5, activation=relu, dense_units=32
    Epoch 1/10
    438/438 - 1s - 3ms/step - accuracy: 0.7747 - loss: 0.0168
    Epoch 2/10
    438/438 - 1s - 2ms/step - accuracy: 0.8374 - loss: 0.0049
    Epoch 3/10
    438/438 - 1s - 2ms/step - accuracy: 0.8665 - loss: 0.0040
    Epoch 4/10
    438/438 - 1s - 2ms/step - accuracy: 0.8830 - loss: 0.0035
    Epoch 5/10
    438/438 - 1s - 2ms/step - accuracy: 0.8922 - loss: 0.0033
    Epoch 6/10
    438/438 - 1s - 2ms/step - accuracy: 0.8973 - loss: 0.0031
    Epoch 7/10
    438/438 - 1s - 2ms/step - accuracy: 0.8994 - loss: 0.0030
    Epoch 8/10
    438/438 - 1s - 2ms/step - accuracy: 0.9010 - loss: 0.0030
    Epoch 9/10
    438/438 - 1s - 2ms/step - accuracy: 0.9027 - loss: 0.0029
    Epoch 10/10
    438/438 - 1s - 2ms/step - accuracy: 0.9020 - loss: 0.0029
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.01331616 0.03247657 0.07385948 0.02191113 
    Mean MAE for this configuration: 0.03539084 

    Training model with filters=32, kernel_size=3x5, activation=relu, dense_units=64
    Epoch 1/10
    438/438 - 2s - 4ms/step - accuracy: 0.7772 - loss: 0.0291
    Epoch 2/10
    438/438 - 1s - 2ms/step - accuracy: 0.8576 - loss: 0.0044
    Epoch 3/10
    438/438 - 1s - 2ms/step - accuracy: 0.8804 - loss: 0.0035
    Epoch 4/10
    438/438 - 1s - 2ms/step - accuracy: 0.8912 - loss: 0.0032
    Epoch 5/10
    438/438 - 1s - 2ms/step - accuracy: 0.8976 - loss: 0.0031
    Epoch 6/10
    438/438 - 1s - 2ms/step - accuracy: 0.9033 - loss: 0.0030
    Epoch 7/10
    438/438 - 1s - 2ms/step - accuracy: 0.9062 - loss: 0.0029
    Epoch 8/10
    438/438 - 1s - 2ms/step - accuracy: 0.9072 - loss: 0.0029
    Epoch 9/10
    438/438 - 1s - 2ms/step - accuracy: 0.9085 - loss: 0.0028
    Epoch 10/10
    438/438 - 1s - 2ms/step - accuracy: 0.9087 - loss: 0.0028
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.01793305 0.03109506 0.07198768 0.01781983 
    Mean MAE for this configuration: 0.0347089 

    Training model with filters=32, kernel_size=3x5, activation=relu, dense_units=128
    Epoch 1/10
    438/438 - 2s - 4ms/step - accuracy: 0.7483 - loss: 0.0496
    Epoch 2/10
    438/438 - 1s - 2ms/step - accuracy: 0.8266 - loss: 0.0157
    Epoch 3/10
    438/438 - 1s - 2ms/step - accuracy: 0.8719 - loss: 0.0041
    Epoch 4/10
    438/438 - 1s - 2ms/step - accuracy: 0.8866 - loss: 0.0035
    Epoch 5/10
    438/438 - 1s - 2ms/step - accuracy: 0.8932 - loss: 0.0032
    Epoch 6/10
    438/438 - 1s - 2ms/step - accuracy: 0.8995 - loss: 0.0031
    Epoch 7/10
    438/438 - 1s - 2ms/step - accuracy: 0.9020 - loss: 0.0030
    Epoch 8/10
    438/438 - 1s - 2ms/step - accuracy: 0.9057 - loss: 0.0029
    Epoch 9/10
    438/438 - 1s - 2ms/step - accuracy: 0.9075 - loss: 0.0029
    Epoch 10/10
    438/438 - 1s - 2ms/step - accuracy: 0.9082 - loss: 0.0028
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.01375671 0.03301882 0.07411041 0.01829217 
    Mean MAE for this configuration: 0.03479453 

    Training model with filters=32, kernel_size=3x5, activation=linear, dense_units=32
    Epoch 1/10
    438/438 - 1s - 3ms/step - accuracy: 0.7392 - loss: 0.0745
    Epoch 2/10
    438/438 - 1s - 2ms/step - accuracy: 0.8461 - loss: 0.0063
    Epoch 3/10
    438/438 - 1s - 1ms/step - accuracy: 0.8711 - loss: 0.0043
    Epoch 4/10
    438/438 - 1s - 1ms/step - accuracy: 0.8774 - loss: 0.0039
    Epoch 5/10
    438/438 - 1s - 1ms/step - accuracy: 0.8806 - loss: 0.0038
    Epoch 6/10
    438/438 - 1s - 2ms/step - accuracy: 0.8827 - loss: 0.0036
    Epoch 7/10
    438/438 - 1s - 2ms/step - accuracy: 0.8832 - loss: 0.0036
    Epoch 8/10
    438/438 - 1s - 2ms/step - accuracy: 0.8839 - loss: 0.0035
    Epoch 9/10
    438/438 - 1s - 2ms/step - accuracy: 0.8832 - loss: 0.0035
    Epoch 10/10
    438/438 - 1s - 1ms/step - accuracy: 0.8832 - loss: 0.0035
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.02193437 0.03398649 0.08244270 0.02995348 
    Mean MAE for this configuration: 0.04207926 

    Training model with filters=32, kernel_size=3x5, activation=linear, dense_units=64
    Epoch 1/10
    438/438 - 2s - 3ms/step - accuracy: 0.8188 - loss: 0.1304
    Epoch 2/10
    438/438 - 1s - 2ms/step - accuracy: 0.8136 - loss: 0.1249
    Epoch 3/10
    438/438 - 1s - 2ms/step - accuracy: 0.7046 - loss: 0.0648
    Epoch 4/10
    438/438 - 1s - 2ms/step - accuracy: 0.8676 - loss: 0.0048
    Epoch 5/10
    438/438 - 1s - 2ms/step - accuracy: 0.8799 - loss: 0.0040
    Epoch 6/10
    438/438 - 1s - 2ms/step - accuracy: 0.8840 - loss: 0.0037
    Epoch 7/10
    438/438 - 1s - 2ms/step - accuracy: 0.8865 - loss: 0.0036
    Epoch 8/10
    438/438 - 1s - 2ms/step - accuracy: 0.8882 - loss: 0.0035
    Epoch 9/10
    438/438 - 1s - 2ms/step - accuracy: 0.8887 - loss: 0.0034
    Epoch 10/10
    438/438 - 1s - 2ms/step - accuracy: 0.8894 - loss: 0.0033
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.01620391 0.03480297 0.07674050 0.02762158 
    Mean MAE for this configuration: 0.03884224 

    Training model with filters=32, kernel_size=3x5, activation=linear, dense_units=128
    Epoch 1/10
    438/438 - 2s - 4ms/step - accuracy: 0.8095 - loss: 0.1317
    Epoch 2/10
    438/438 - 1s - 2ms/step - accuracy: 0.8228 - loss: 0.1288
    Epoch 3/10
    438/438 - 1s - 2ms/step - accuracy: 0.7943 - loss: 0.1265
    Epoch 4/10
    438/438 - 1s - 2ms/step - accuracy: 0.6958 - loss: 0.1099
    Epoch 5/10
    438/438 - 1s - 2ms/step - accuracy: 0.7070 - loss: 0.0658
    Epoch 6/10
    438/438 - 1s - 2ms/step - accuracy: 0.8315 - loss: 0.0424
    Epoch 7/10
    438/438 - 1s - 2ms/step - accuracy: 0.8216 - loss: 0.0424
    Epoch 8/10
    438/438 - 1s - 2ms/step - accuracy: 0.8650 - loss: 0.0422
    Epoch 9/10
    438/438 - 1s - 2ms/step - accuracy: 0.8746 - loss: 0.0422
    Epoch 10/10
    438/438 - 1s - 2ms/step - accuracy: 0.8632 - loss: 0.0423
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.20931609 0.04285909 0.29922203 0.02822976 
    Mean MAE for this configuration: 0.1449067 

    Training model with filters=64, kernel_size=3x3, activation=relu, dense_units=32
    Epoch 1/10
    438/438 - 2s - 4ms/step - accuracy: 0.7858 - loss: 0.0153
    Epoch 2/10
    438/438 - 1s - 2ms/step - accuracy: 0.8425 - loss: 0.0043
    Epoch 3/10
    438/438 - 1s - 2ms/step - accuracy: 0.8651 - loss: 0.0037
    Epoch 4/10
    438/438 - 1s - 2ms/step - accuracy: 0.8822 - loss: 0.0033
    Epoch 5/10
    438/438 - 1s - 2ms/step - accuracy: 0.8879 - loss: 0.0031
    Epoch 6/10
    438/438 - 1s - 2ms/step - accuracy: 0.8930 - loss: 0.0031
    Epoch 7/10
    438/438 - 1s - 2ms/step - accuracy: 0.8941 - loss: 0.0030
    Epoch 8/10
    438/438 - 1s - 2ms/step - accuracy: 0.8967 - loss: 0.0030
    Epoch 9/10
    438/438 - 1s - 2ms/step - accuracy: 0.8992 - loss: 0.0030
    Epoch 10/10
    438/438 - 1s - 2ms/step - accuracy: 0.9016 - loss: 0.0029
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.02070135 0.03151289 0.07209245 0.02176060 
    Mean MAE for this configuration: 0.03651682 

    Training model with filters=64, kernel_size=3x3, activation=relu, dense_units=64
    Epoch 1/10
    438/438 - 2s - 4ms/step - accuracy: 0.7284 - loss: 0.0336
    Epoch 2/10
    438/438 - 1s - 2ms/step - accuracy: 0.8172 - loss: 0.0077
    Epoch 3/10
    438/438 - 1s - 2ms/step - accuracy: 0.8545 - loss: 0.0041
    Epoch 4/10
    438/438 - 1s - 2ms/step - accuracy: 0.8734 - loss: 0.0036
    Epoch 5/10
    438/438 - 1s - 2ms/step - accuracy: 0.8800 - loss: 0.0034
    Epoch 6/10
    438/438 - 1s - 2ms/step - accuracy: 0.8897 - loss: 0.0032
    Epoch 7/10
    438/438 - 1s - 2ms/step - accuracy: 0.8914 - loss: 0.0031
    Epoch 8/10
    438/438 - 1s - 2ms/step - accuracy: 0.8932 - loss: 0.0030
    Epoch 9/10
    438/438 - 1s - 2ms/step - accuracy: 0.8944 - loss: 0.0030
    Epoch 10/10
    438/438 - 1s - 2ms/step - accuracy: 0.8962 - loss: 0.0029
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.01922012 0.03270522 0.07911101 0.02482758 
    Mean MAE for this configuration: 0.03896598 

    Training model with filters=64, kernel_size=3x3, activation=relu, dense_units=128
    Epoch 1/10
    438/438 - 2s - 5ms/step - accuracy: 0.7498 - loss: 0.0519
    Epoch 2/10
    438/438 - 1s - 2ms/step - accuracy: 0.8263 - loss: 0.0237
    Epoch 3/10
    438/438 - 1s - 2ms/step - accuracy: 0.8749 - loss: 0.0038
    Epoch 4/10
    438/438 - 1s - 2ms/step - accuracy: 0.8854 - loss: 0.0033
    Epoch 5/10
    438/438 - 1s - 2ms/step - accuracy: 0.8880 - loss: 0.0032
    Epoch 6/10
    438/438 - 1s - 2ms/step - accuracy: 0.8927 - loss: 0.0031
    Epoch 7/10
    438/438 - 1s - 2ms/step - accuracy: 0.8949 - loss: 0.0030
    Epoch 8/10
    438/438 - 1s - 2ms/step - accuracy: 0.8974 - loss: 0.0030
    Epoch 9/10
    438/438 - 1s - 2ms/step - accuracy: 0.9010 - loss: 0.0029
    Epoch 10/10
    438/438 - 1s - 2ms/step - accuracy: 0.9060 - loss: 0.0029
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.01247352 0.03307977 0.07320733 0.02285556 
    Mean MAE for this configuration: 0.03540405 

    Training model with filters=64, kernel_size=3x3, activation=linear, dense_units=32
    Epoch 1/10
    438/438 - 2s - 4ms/step - accuracy: 0.8193 - loss: 0.1430
    Epoch 2/10
    438/438 - 1s - 2ms/step - accuracy: 0.8199 - loss: 0.1434
    Epoch 3/10
    438/438 - 1s - 2ms/step - accuracy: 0.8211 - loss: 0.1413
    Epoch 4/10
    438/438 - 1s - 2ms/step - accuracy: 0.8130 - loss: 0.1415
    Epoch 5/10
    438/438 - 1s - 2ms/step - accuracy: 0.8173 - loss: 0.1361
    Epoch 6/10
    438/438 - 1s - 2ms/step - accuracy: 0.8201 - loss: 0.1434
    Epoch 7/10
    438/438 - 1s - 2ms/step - accuracy: 0.8201 - loss: 0.1435
    Epoch 8/10
    438/438 - 1s - 2ms/step - accuracy: 0.8201 - loss: 0.1434
    Epoch 9/10
    438/438 - 1s - 2ms/step - accuracy: 0.8201 - loss: 0.1434
    Epoch 10/10
    438/438 - 1s - 2ms/step - accuracy: 0.8201 - loss: 0.1434
    188/188 - 0s - 1ms/step
       preval     crate     ptran      prec 
    0.2094271 0.3790987 0.2992234 0.4993451 
    Mean MAE for this configuration: 0.3467736 

    Training model with filters=64, kernel_size=3x3, activation=linear, dense_units=64
    Epoch 1/10
    438/438 - 2s - 4ms/step - accuracy: 0.8168 - loss: 0.1298
    Epoch 2/10
    438/438 - 1s - 2ms/step - accuracy: 0.8171 - loss: 0.1299
    Epoch 3/10
    438/438 - 1s - 2ms/step - accuracy: 0.8210 - loss: 0.1275
    Epoch 4/10
    438/438 - 1s - 2ms/step - accuracy: 0.8232 - loss: 0.1266
    Epoch 5/10
    438/438 - 1s - 2ms/step - accuracy: 0.8147 - loss: 0.1289
    Epoch 6/10
    438/438 - 1s - 2ms/step - accuracy: 0.8226 - loss: 0.1261
    Epoch 7/10
    438/438 - 1s - 2ms/step - accuracy: 0.8225 - loss: 0.1261
    Epoch 8/10
    438/438 - 1s - 2ms/step - accuracy: 0.8178 - loss: 0.1253
    Epoch 9/10
    438/438 - 1s - 2ms/step - accuracy: 0.8263 - loss: 0.1245
    Epoch 10/10
    438/438 - 1s - 2ms/step - accuracy: 0.8263 - loss: 0.1248
    188/188 - 0s - 1ms/step
       preval     crate     ptran      prec 
    0.2094269 0.3787076 0.2992168 0.4118670 
    Mean MAE for this configuration: 0.3248046 

    Training model with filters=64, kernel_size=3x3, activation=linear, dense_units=128
    Epoch 1/10
    438/438 - 2s - 4ms/step - accuracy: 0.8118 - loss: 0.1404
    Epoch 2/10
    438/438 - 1s - 2ms/step - accuracy: 0.8172 - loss: 0.1419
    Epoch 3/10
    438/438 - 1s - 2ms/step - accuracy: 0.8210 - loss: 0.1352
    Epoch 4/10
    438/438 - 1s - 2ms/step - accuracy: 0.8198 - loss: 0.1397
    Epoch 5/10
    438/438 - 1s - 2ms/step - accuracy: 0.8199 - loss: 0.1419
    Epoch 6/10
    438/438 - 1s - 2ms/step - accuracy: 0.8198 - loss: 0.1418
    Epoch 7/10
    438/438 - 1s - 2ms/step - accuracy: 0.8198 - loss: 0.1372
    Epoch 8/10
    438/438 - 1s - 2ms/step - accuracy: 0.8198 - loss: 0.1368
    Epoch 9/10
    438/438 - 1s - 2ms/step - accuracy: 0.8187 - loss: 0.1406
    Epoch 10/10
    438/438 - 1s - 2ms/step - accuracy: 0.8198 - loss: 0.1388
    188/188 - 0s - 1ms/step
       preval     crate     ptran      prec 
    0.2094271 0.3791292 0.2992234 0.4739503 
    Mean MAE for this configuration: 0.3404325 

    Training model with filters=64, kernel_size=5x5, activation=relu, dense_units=32
    Epoch 1/10
    438/438 - 2s - 4ms/step - accuracy: 0.7913 - loss: 0.0110
    Epoch 2/10
    438/438 - 1s - 2ms/step - accuracy: 0.8586 - loss: 0.0042
    Epoch 3/10
    438/438 - 1s - 2ms/step - accuracy: 0.8790 - loss: 0.0035
    Epoch 4/10
    438/438 - 1s - 2ms/step - accuracy: 0.8921 - loss: 0.0032
    Epoch 5/10
    438/438 - 1s - 2ms/step - accuracy: 0.8979 - loss: 0.0031
    Epoch 6/10
    438/438 - 1s - 2ms/step - accuracy: 0.8979 - loss: 0.0030
    Epoch 7/10
    438/438 - 1s - 2ms/step - accuracy: 0.9026 - loss: 0.0029
    Epoch 8/10
    438/438 - 1s - 2ms/step - accuracy: 0.9035 - loss: 0.0029
    Epoch 9/10
    438/438 - 1s - 2ms/step - accuracy: 0.9073 - loss: 0.0028
    Epoch 10/10
    438/438 - 1s - 2ms/step - accuracy: 0.9073 - loss: 0.0028
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.01333550 0.03212444 0.07332928 0.01838902 
    Mean MAE for this configuration: 0.03429456 

    Training model with filters=64, kernel_size=5x5, activation=relu, dense_units=64
    Epoch 1/10
    438/438 - 2s - 4ms/step - accuracy: 0.7279 - loss: 0.0469
    Epoch 2/10
    438/438 - 1s - 2ms/step - accuracy: 0.8372 - loss: 0.0051
    Epoch 3/10
    438/438 - 1s - 2ms/step - accuracy: 0.8779 - loss: 0.0036
    Epoch 4/10
    438/438 - 1s - 2ms/step - accuracy: 0.8880 - loss: 0.0032
    Epoch 5/10
    438/438 - 1s - 2ms/step - accuracy: 0.8932 - loss: 0.0031
    Epoch 6/10
    438/438 - 1s - 2ms/step - accuracy: 0.8969 - loss: 0.0030
    Epoch 7/10
    438/438 - 1s - 2ms/step - accuracy: 0.8990 - loss: 0.0029
    Epoch 8/10
    438/438 - 1s - 2ms/step - accuracy: 0.9041 - loss: 0.0029
    Epoch 9/10
    438/438 - 1s - 2ms/step - accuracy: 0.9055 - loss: 0.0028
    Epoch 10/10
    438/438 - 1s - 2ms/step - accuracy: 0.9054 - loss: 0.0028
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.01249961 0.03247475 0.07375686 0.02032560 
    Mean MAE for this configuration: 0.03476421 

    Training model with filters=64, kernel_size=5x5, activation=relu, dense_units=128
    Epoch 1/10
    438/438 - 2s - 4ms/step - accuracy: 0.7755 - loss: 0.0191
    Epoch 2/10
    438/438 - 1s - 2ms/step - accuracy: 0.8686 - loss: 0.0039
    Epoch 3/10
    438/438 - 1s - 2ms/step - accuracy: 0.8864 - loss: 0.0033
    Epoch 4/10
    438/438 - 1s - 2ms/step - accuracy: 0.8925 - loss: 0.0031
    Epoch 5/10
    438/438 - 1s - 2ms/step - accuracy: 0.9001 - loss: 0.0030
    Epoch 6/10
    438/438 - 1s - 2ms/step - accuracy: 0.9025 - loss: 0.0029
    Epoch 7/10
    438/438 - 1s - 2ms/step - accuracy: 0.9036 - loss: 0.0029
    Epoch 8/10
    438/438 - 1s - 2ms/step - accuracy: 0.9070 - loss: 0.0028
    Epoch 9/10
    438/438 - 1s - 2ms/step - accuracy: 0.9076 - loss: 0.0028
    Epoch 10/10
    438/438 - 1s - 2ms/step - accuracy: 0.9078 - loss: 0.0027
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.01456389 0.03451676 0.07373209 0.01703268 
    Mean MAE for this configuration: 0.03496136 

    Training model with filters=64, kernel_size=5x5, activation=linear, dense_units=32
    Epoch 1/10
    438/438 - 2s - 4ms/step - accuracy: 0.7533 - loss: 0.0635
    Epoch 2/10
    438/438 - 1s - 2ms/step - accuracy: 0.8369 - loss: 0.0058
    Epoch 3/10
    438/438 - 1s - 2ms/step - accuracy: 0.8621 - loss: 0.0042
    Epoch 4/10
    438/438 - 1s - 2ms/step - accuracy: 0.8737 - loss: 0.0038
    Epoch 5/10
    438/438 - 1s - 2ms/step - accuracy: 0.8788 - loss: 0.0036
    Epoch 6/10
    438/438 - 1s - 2ms/step - accuracy: 0.8797 - loss: 0.0035
    Epoch 7/10
    438/438 - 1s - 2ms/step - accuracy: 0.8814 - loss: 0.0035
    Epoch 8/10
    438/438 - 1s - 2ms/step - accuracy: 0.8829 - loss: 0.0034
    Epoch 9/10
    438/438 - 1s - 2ms/step - accuracy: 0.8837 - loss: 0.0034
    Epoch 10/10
    438/438 - 1s - 2ms/step - accuracy: 0.8856 - loss: 0.0034
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.01876336 0.03319909 0.08052284 0.03108091 
    Mean MAE for this configuration: 0.04089155 

    Training model with filters=64, kernel_size=5x5, activation=linear, dense_units=64
    Epoch 1/10
    438/438 - 2s - 4ms/step - accuracy: 0.8143 - loss: 0.1325
    Epoch 2/10
    438/438 - 1s - 2ms/step - accuracy: 0.7929 - loss: 0.0309
    Epoch 3/10
    438/438 - 1s - 2ms/step - accuracy: 0.8714 - loss: 0.0042
    Epoch 4/10
    438/438 - 1s - 2ms/step - accuracy: 0.8814 - loss: 0.0038
    Epoch 5/10
    438/438 - 1s - 2ms/step - accuracy: 0.8835 - loss: 0.0036
    Epoch 6/10
    438/438 - 1s - 2ms/step - accuracy: 0.8822 - loss: 0.0035
    Epoch 7/10
    438/438 - 1s - 2ms/step - accuracy: 0.8842 - loss: 0.0035
    Epoch 8/10
    438/438 - 1s - 2ms/step - accuracy: 0.8852 - loss: 0.0034
    Epoch 9/10
    438/438 - 1s - 2ms/step - accuracy: 0.8881 - loss: 0.0034
    Epoch 10/10
    438/438 - 1s - 2ms/step - accuracy: 0.8893 - loss: 0.0033
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.02334054 0.03512780 0.07785099 0.02664145 
    Mean MAE for this configuration: 0.04074019 

    Training model with filters=64, kernel_size=5x5, activation=linear, dense_units=128
    Epoch 1/10
    438/438 - 2s - 4ms/step - accuracy: 0.8253 - loss: 0.1339
    Epoch 2/10
    438/438 - 1s - 2ms/step - accuracy: 0.8203 - loss: 0.1326
    Epoch 3/10
    438/438 - 1s - 2ms/step - accuracy: 0.8201 - loss: 0.1391
    Epoch 4/10
    438/438 - 1s - 2ms/step - accuracy: 0.8201 - loss: 0.1434
    Epoch 5/10
    438/438 - 1s - 2ms/step - accuracy: 0.8201 - loss: 0.1434
    Epoch 6/10
    438/438 - 1s - 2ms/step - accuracy: 0.8201 - loss: 0.1434
    Epoch 7/10
    438/438 - 1s - 2ms/step - accuracy: 0.8201 - loss: 0.1434
    Epoch 8/10
    438/438 - 1s - 2ms/step - accuracy: 0.8201 - loss: 0.1434
    Epoch 9/10
    438/438 - 1s - 2ms/step - accuracy: 0.8201 - loss: 0.1434
    Epoch 10/10
    438/438 - 1s - 2ms/step - accuracy: 0.8201 - loss: 0.1434
    188/188 - 0s - 1ms/step
       preval     crate     ptran      prec 
    0.2094271 0.3790988 0.2992234 0.4993458 
    Mean MAE for this configuration: 0.3467738 

    Training model with filters=64, kernel_size=3x5, activation=relu, dense_units=32
    Epoch 1/10
    438/438 - 2s - 4ms/step - accuracy: 0.8074 - loss: 0.0118
    Epoch 2/10
    438/438 - 1s - 2ms/step - accuracy: 0.8599 - loss: 0.0045
    Epoch 3/10
    438/438 - 1s - 2ms/step - accuracy: 0.8781 - loss: 0.0037
    Epoch 4/10
    438/438 - 1s - 2ms/step - accuracy: 0.8825 - loss: 0.0034
    Epoch 5/10
    438/438 - 1s - 2ms/step - accuracy: 0.8873 - loss: 0.0032
    Epoch 6/10
    438/438 - 1s - 2ms/step - accuracy: 0.8889 - loss: 0.0031
    Epoch 7/10
    438/438 - 1s - 2ms/step - accuracy: 0.8934 - loss: 0.0031
    Epoch 8/10
    438/438 - 1s - 2ms/step - accuracy: 0.8952 - loss: 0.0030
    Epoch 9/10
    438/438 - 1s - 2ms/step - accuracy: 0.8976 - loss: 0.0030
    Epoch 10/10
    438/438 - 1s - 2ms/step - accuracy: 0.8973 - loss: 0.0029
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.01279331 0.03201401 0.07628702 0.02195548 
    Mean MAE for this configuration: 0.03576245 

    Training model with filters=64, kernel_size=3x5, activation=relu, dense_units=64
    Epoch 1/10
    438/438 - 2s - 4ms/step - accuracy: 0.7356 - loss: 0.0309
    Epoch 2/10
    438/438 - 1s - 2ms/step - accuracy: 0.8527 - loss: 0.0047
    Epoch 3/10
    438/438 - 1s - 2ms/step - accuracy: 0.8796 - loss: 0.0037
    Epoch 4/10
    438/438 - 1s - 2ms/step - accuracy: 0.8898 - loss: 0.0033
    Epoch 5/10
    438/438 - 1s - 2ms/step - accuracy: 0.8951 - loss: 0.0031
    Epoch 6/10
    438/438 - 1s - 2ms/step - accuracy: 0.8995 - loss: 0.0030
    Epoch 7/10
    438/438 - 1s - 2ms/step - accuracy: 0.9017 - loss: 0.0030
    Epoch 8/10
    438/438 - 1s - 2ms/step - accuracy: 0.8999 - loss: 0.0029
    Epoch 9/10
    438/438 - 1s - 2ms/step - accuracy: 0.9047 - loss: 0.0029
    Epoch 10/10
    438/438 - 1s - 2ms/step - accuracy: 0.9055 - loss: 0.0029
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.01882770 0.03273035 0.07514253 0.02023832 
    Mean MAE for this configuration: 0.03673472 

    Training model with filters=64, kernel_size=3x5, activation=relu, dense_units=128
    Epoch 1/10
    438/438 - 2s - 5ms/step - accuracy: 0.7890 - loss: 0.0179
    Epoch 2/10
    438/438 - 1s - 2ms/step - accuracy: 0.8599 - loss: 0.0041
    Epoch 3/10
    438/438 - 1s - 2ms/step - accuracy: 0.8732 - loss: 0.0035
    Epoch 4/10
    438/438 - 1s - 2ms/step - accuracy: 0.8821 - loss: 0.0032
    Epoch 5/10
    438/438 - 1s - 2ms/step - accuracy: 0.8900 - loss: 0.0031
    Epoch 6/10
    438/438 - 1s - 2ms/step - accuracy: 0.8969 - loss: 0.0030
    Epoch 7/10
    438/438 - 1s - 2ms/step - accuracy: 0.9022 - loss: 0.0029
    Epoch 8/10
    438/438 - 1s - 2ms/step - accuracy: 0.9038 - loss: 0.0029
    Epoch 9/10
    438/438 - 1s - 2ms/step - accuracy: 0.9040 - loss: 0.0028
    Epoch 10/10
    438/438 - 1s - 2ms/step - accuracy: 0.9071 - loss: 0.0028
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.01443516 0.03302710 0.07488399 0.02000989 
    Mean MAE for this configuration: 0.03558903 

    Training model with filters=64, kernel_size=3x5, activation=linear, dense_units=32
    Epoch 1/10
    438/438 - 2s - 4ms/step - accuracy: 0.7773 - loss: 0.0965
    Epoch 2/10
    438/438 - 1s - 2ms/step - accuracy: 0.8315 - loss: 0.0062
    Epoch 3/10
    438/438 - 1s - 2ms/step - accuracy: 0.8714 - loss: 0.0041
    Epoch 4/10
    438/438 - 1s - 2ms/step - accuracy: 0.8803 - loss: 0.0038
    Epoch 5/10
    438/438 - 1s - 2ms/step - accuracy: 0.8832 - loss: 0.0036
    Epoch 6/10
    438/438 - 1s - 2ms/step - accuracy: 0.8832 - loss: 0.0035
    Epoch 7/10
    438/438 - 1s - 2ms/step - accuracy: 0.8834 - loss: 0.0034
    Epoch 8/10
    438/438 - 1s - 2ms/step - accuracy: 0.8857 - loss: 0.0034
    Epoch 9/10
    438/438 - 1s - 2ms/step - accuracy: 0.8914 - loss: 0.0033
    Epoch 10/10
    438/438 - 1s - 2ms/step - accuracy: 0.8909 - loss: 0.0033
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.01915425 0.03289284 0.07924669 0.02778440 
    Mean MAE for this configuration: 0.03976954 

    Training model with filters=64, kernel_size=3x5, activation=linear, dense_units=64
    Epoch 1/10
    438/438 - 2s - 4ms/step - accuracy: 0.8144 - loss: 0.1288
    Epoch 2/10
    438/438 - 1s - 2ms/step - accuracy: 0.8168 - loss: 0.1207
    Epoch 3/10
    438/438 - 1s - 2ms/step - accuracy: 0.7825 - loss: 0.0252
    Epoch 4/10
    438/438 - 1s - 2ms/step - accuracy: 0.8744 - loss: 0.0041
    Epoch 5/10
    438/438 - 1s - 2ms/step - accuracy: 0.8795 - loss: 0.0037
    Epoch 6/10
    438/438 - 1s - 2ms/step - accuracy: 0.8854 - loss: 0.0035
    Epoch 7/10
    438/438 - 1s - 2ms/step - accuracy: 0.8866 - loss: 0.0034
    Epoch 8/10
    438/438 - 1s - 2ms/step - accuracy: 0.8869 - loss: 0.0034
    Epoch 9/10
    438/438 - 1s - 2ms/step - accuracy: 0.8876 - loss: 0.0034
    Epoch 10/10
    438/438 - 1s - 2ms/step - accuracy: 0.8857 - loss: 0.0033
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.02117387 0.03577308 0.07521690 0.03079652 
    Mean MAE for this configuration: 0.0407401 

    Training model with filters=64, kernel_size=3x5, activation=linear, dense_units=128
    Epoch 1/10
    438/438 - 2s - 4ms/step - accuracy: 0.8124 - loss: 0.1377
    Epoch 2/10
    438/438 - 1s - 2ms/step - accuracy: 0.8201 - loss: 0.1429
    Epoch 3/10
    438/438 - 1s - 2ms/step - accuracy: 0.8201 - loss: 0.1434
    Epoch 4/10
    438/438 - 1s - 2ms/step - accuracy: 0.8201 - loss: 0.1434
    Epoch 5/10
    438/438 - 1s - 2ms/step - accuracy: 0.8201 - loss: 0.1434
    Epoch 6/10
    438/438 - 1s - 2ms/step - accuracy: 0.8201 - loss: 0.1434
    Epoch 7/10
    438/438 - 1s - 2ms/step - accuracy: 0.8201 - loss: 0.1434
    Epoch 8/10
    438/438 - 1s - 2ms/step - accuracy: 0.8201 - loss: 0.1434
    Epoch 9/10
    438/438 - 1s - 2ms/step - accuracy: 0.8201 - loss: 0.1434
    Epoch 10/10
    438/438 - 1s - 2ms/step - accuracy: 0.8201 - loss: 0.1434
    188/188 - 0s - 1ms/step
       preval     crate     ptran      prec 
    0.2094270 0.3790987 0.2992234 0.4993458 
    Mean MAE for this configuration: 0.3467737 

``` r
# Display the results of all configurations and their MAEs
print(results)
```

        filters kernel_size activation dense_units        MAE
     1:      16         3x3       relu          32 0.03752825
     2:      16         3x3       relu          64 0.03960608
     3:      16         3x3       relu         128 0.03663024
     4:      16         3x3     linear          32 0.04148222
     5:      16         3x3     linear          64 0.04517099
     6:      16         3x3     linear         128 0.15322776
     7:      16         5x5       relu          32 0.03680819
     8:      16         5x5       relu          64 0.03570932
     9:      16         5x5       relu         128 0.03613636
    10:      16         5x5     linear          32 0.04225757
    11:      16         5x5     linear          64 0.04425998
    12:      16         5x5     linear         128 0.04012672
    13:      16         3x5       relu          32 0.03828017
    14:      16         3x5       relu          64 0.03733097
    15:      16         3x5       relu         128 0.03605900
    16:      16         3x5     linear          32 0.04364775
    17:      16         3x5     linear          64 0.04346280
    18:      16         3x5     linear         128 0.04197080
    19:      32         3x3       relu          32 0.03768336
    20:      32         3x3       relu          64 0.03684318
    21:      32         3x3       relu         128 0.03776012
    22:      32         3x3     linear          32 0.04116954
    23:      32         3x3     linear          64 0.22943593
    24:      32         3x3     linear         128 0.33825079
    25:      32         5x5       relu          32 0.03541435
    26:      32         5x5       relu          64 0.03687961
    27:      32         5x5       relu         128 0.03589573
    28:      32         5x5     linear          32 0.04234969
    29:      32         5x5     linear          64 0.04009520
    30:      32         5x5     linear         128 0.04359647
    31:      32         3x5       relu          32 0.03539084
    32:      32         3x5       relu          64 0.03470890
    33:      32         3x5       relu         128 0.03479453
    34:      32         3x5     linear          32 0.04207926
    35:      32         3x5     linear          64 0.03884224
    36:      32         3x5     linear         128 0.14490674
    37:      64         3x3       relu          32 0.03651682
    38:      64         3x3       relu          64 0.03896598
    39:      64         3x3       relu         128 0.03540405
    40:      64         3x3     linear          32 0.34677357
    41:      64         3x3     linear          64 0.32480458
    42:      64         3x3     linear         128 0.34043250
    43:      64         5x5       relu          32 0.03429456
    44:      64         5x5       relu          64 0.03476421
    45:      64         5x5       relu         128 0.03496136
    46:      64         5x5     linear          32 0.04089155
    47:      64         5x5     linear          64 0.04074019
    48:      64         5x5     linear         128 0.34677376
    49:      64         3x5       relu          32 0.03576245
    50:      64         3x5       relu          64 0.03673472
    51:      64         3x5       relu         128 0.03558903
    52:      64         3x5     linear          32 0.03976954
    53:      64         3x5     linear          64 0.04074010
    54:      64         3x5     linear         128 0.34677374
        filters kernel_size activation dense_units        MAE

``` r
# Find the best model configuration
find_best_model <- function(results) {
  # Select the row with the minimum MAE
  best_model <- results[which.min(MAE)]
  return(best_model)
}

# Train and evaluate the best model
train_and_evaluate_best_model <- function(best_model, train_data, test_data, theta, input_shape, output_units, epochs = 10) {
  # Extract the best hyperparameters
  best_filters <- best_model$filters
  best_kernel_size <- as.integer(unlist(strsplit(best_model$kernel_size, "x")))
  best_activation <- best_model$activation
  best_dense_units <- best_model$dense_units
  
  # Build the best model
  model <- build_cnn_model(
    input_shape = input_shape,
    output_units = output_units,
    filters = best_filters,
    kernel_size = best_kernel_size,
    activation = best_activation,
    dense_units = best_dense_units
  )
  
  # Train the best model
  train_model(model, train_data, epochs = epochs)
  
  # Evaluate the best model
  eval_results <- evaluate_model(model, test_data, theta)
  return(list(model = model, eval_results = eval_results))
}
```

``` r
plot_results <- function(pred, test_data, theta, MAEs, N, N_train) {
  # Prepare the data for plotting
  pred[, id := 1L:.N]
  pred[, crate := qlogis(crate) * 10]
  pred_long <- melt(pred, id.vars = "id")
  
  theta_long <- test_data$y |> as.data.table()
  setnames(theta_long, names(theta))
  theta_long[, id := 1L:.N]
  theta_long[, crate := qlogis(crate) * 10]
  theta_long <- melt(theta_long, id.vars = "id")
  
  alldat <- rbind(
    cbind(pred_long, Type = "Predicted"),
    cbind(theta_long, Type = "Observed")
  )
  
  # Plot 1: Boxplot of Predicted vs Observed values
  p1 <- ggplot(alldat, aes(x = value, colour = Type)) +
    facet_wrap(~variable, scales = "free") +
    geom_boxplot() +
    labs(title = "Boxplot: Predicted vs Observed")
  
  print(p1)  # Display the first plot
  
  # Prepare data for second plot
  alldat_wide <- dcast(alldat, id + variable ~ Type, value.var = "value")
  
  vnames <- data.table(
    variable = c("preval", "crate", "ptran", "prec"),
    Name     = paste(
      c("Init. state", "Contact Rate", "P(transmit)", "P(recover)"),
      sprintf("(MAE: %.2f)", MAEs)
    )
  )
  
  alldat_wide <- merge(alldat_wide, vnames, by = "variable")
  
  # Plot 2: Observed vs Predicted with MAE labels
  p2 <- ggplot(alldat_wide, aes(x = Observed, y = Predicted)) +
    facet_wrap(~ Name, scales = "free") +
    geom_abline(slope = 1, intercept = 0) +
    geom_point(alpha = .2) +
    labs(
      title    = "Observed vs Predicted (validation set)",
      subtitle = sprintf(
        "The model includes %i simulated datasets, of which %i were used for training.",
        N, N_train
      ),
      caption  = "Predictions made using a CNN as implemented with loss function MAE."
    )
  
  print(p2)  # Display the second plot
}
```

``` r
best_model <- find_best_model(results)

print(best_model)
```

       filters kernel_size activation dense_units        MAE
    1:      64         5x5       relu          32 0.03429456

``` r
best_model_results <- train_and_evaluate_best_model(best_model, train, test, theta2, input_shape, output_units)
```

    Epoch 1/10
    438/438 - 2s - 4ms/step - accuracy: 0.7913 - loss: 0.0110
    Epoch 2/10
    438/438 - 1s - 2ms/step - accuracy: 0.8586 - loss: 0.0042
    Epoch 3/10
    438/438 - 1s - 2ms/step - accuracy: 0.8790 - loss: 0.0035
    Epoch 4/10
    438/438 - 1s - 2ms/step - accuracy: 0.8921 - loss: 0.0032
    Epoch 5/10
    438/438 - 1s - 2ms/step - accuracy: 0.8979 - loss: 0.0031
    Epoch 6/10
    438/438 - 1s - 2ms/step - accuracy: 0.8979 - loss: 0.0030
    Epoch 7/10
    438/438 - 1s - 2ms/step - accuracy: 0.9026 - loss: 0.0029
    Epoch 8/10
    438/438 - 1s - 2ms/step - accuracy: 0.9035 - loss: 0.0029
    Epoch 9/10
    438/438 - 1s - 2ms/step - accuracy: 0.9073 - loss: 0.0028
    Epoch 10/10
    438/438 - 1s - 2ms/step - accuracy: 0.9073 - loss: 0.0028
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.01333550 0.03212444 0.07332928 0.01838902 

``` r
pred <- best_model_results$eval_results$pred
MAEs <- best_model_results$eval_results$MAEs
print(MAEs)
```

        preval      crate      ptran       prec 
    0.01333550 0.03212444 0.07332928 0.01838902 

``` r
plot_results(pred, test, theta2, MAEs, N, floor(N * 0.7))
```

![](CNN_SIR_all_files/figure-commonmark/plotting%20the%20best%20results-1.png)

![](CNN_SIR_all_files/figure-commonmark/plotting%20the%20best%20results-2.png)
