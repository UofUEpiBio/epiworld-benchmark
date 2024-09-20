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
    438/438 - 1s - 3ms/step - accuracy: 0.7161 - loss: 0.0390
    Epoch 2/100
    438/438 - 1s - 1ms/step - accuracy: 0.7998 - loss: 0.0089
    Epoch 3/100
    438/438 - 1s - 1ms/step - accuracy: 0.8428 - loss: 0.0056
    Epoch 4/100
    438/438 - 1s - 1ms/step - accuracy: 0.8634 - loss: 0.0046
    Epoch 5/100
    438/438 - 1s - 1ms/step - accuracy: 0.8721 - loss: 0.0041
    Epoch 6/100
    438/438 - 1s - 1ms/step - accuracy: 0.8789 - loss: 0.0039
    Epoch 7/100
    438/438 - 1s - 1ms/step - accuracy: 0.8811 - loss: 0.0037
    Epoch 8/100
    438/438 - 1s - 1ms/step - accuracy: 0.8837 - loss: 0.0036
    Epoch 9/100
    438/438 - 1s - 1ms/step - accuracy: 0.8851 - loss: 0.0035
    Epoch 10/100
    438/438 - 1s - 1ms/step - accuracy: 0.8869 - loss: 0.0034
    Epoch 11/100
    438/438 - 1s - 1ms/step - accuracy: 0.8893 - loss: 0.0034
    Epoch 12/100
    438/438 - 1s - 1ms/step - accuracy: 0.8904 - loss: 0.0034
    Epoch 13/100
    438/438 - 1s - 1ms/step - accuracy: 0.8919 - loss: 0.0033
    Epoch 14/100
    438/438 - 1s - 1ms/step - accuracy: 0.8920 - loss: 0.0033
    Epoch 15/100
    438/438 - 1s - 1ms/step - accuracy: 0.8947 - loss: 0.0032
    Epoch 16/100
    438/438 - 1s - 1ms/step - accuracy: 0.8937 - loss: 0.0032
    Epoch 17/100
    438/438 - 1s - 1ms/step - accuracy: 0.8931 - loss: 0.0033
    Epoch 18/100
    438/438 - 1s - 1ms/step - accuracy: 0.8938 - loss: 0.0032
    Epoch 19/100
    438/438 - 1s - 1ms/step - accuracy: 0.8919 - loss: 0.0032
    Epoch 20/100
    438/438 - 1s - 1ms/step - accuracy: 0.8900 - loss: 0.0032
    Epoch 21/100
    438/438 - 1s - 1ms/step - accuracy: 0.8914 - loss: 0.0032
    Epoch 22/100
    438/438 - 1s - 1ms/step - accuracy: 0.8925 - loss: 0.0032
    Epoch 23/100
    438/438 - 1s - 1ms/step - accuracy: 0.8919 - loss: 0.0032
    Epoch 24/100
    438/438 - 1s - 1ms/step - accuracy: 0.8917 - loss: 0.0032
    Epoch 25/100
    438/438 - 1s - 1ms/step - accuracy: 0.8917 - loss: 0.0032
    Epoch 26/100
    438/438 - 1s - 1ms/step - accuracy: 0.8927 - loss: 0.0032
    Epoch 27/100
    438/438 - 1s - 1ms/step - accuracy: 0.8930 - loss: 0.0032
    Epoch 28/100
    438/438 - 1s - 1ms/step - accuracy: 0.8939 - loss: 0.0032
    Epoch 29/100
    438/438 - 1s - 1ms/step - accuracy: 0.8929 - loss: 0.0032
    Epoch 30/100
    438/438 - 1s - 1ms/step - accuracy: 0.8935 - loss: 0.0032
    Epoch 31/100
    438/438 - 1s - 1ms/step - accuracy: 0.8938 - loss: 0.0032
    Epoch 32/100
    438/438 - 1s - 1ms/step - accuracy: 0.8957 - loss: 0.0032
    Epoch 33/100
    438/438 - 1s - 1ms/step - accuracy: 0.8928 - loss: 0.0032
    Epoch 34/100
    438/438 - 1s - 1ms/step - accuracy: 0.8927 - loss: 0.0032
    Epoch 35/100
    438/438 - 1s - 1ms/step - accuracy: 0.8946 - loss: 0.0032
    Epoch 36/100
    438/438 - 1s - 1ms/step - accuracy: 0.8944 - loss: 0.0032
    Epoch 37/100
    438/438 - 1s - 1ms/step - accuracy: 0.8959 - loss: 0.0032
    Epoch 38/100
    438/438 - 1s - 1ms/step - accuracy: 0.8929 - loss: 0.0032
    Epoch 39/100
    438/438 - 1s - 1ms/step - accuracy: 0.8958 - loss: 0.0032
    Epoch 40/100
    438/438 - 1s - 1ms/step - accuracy: 0.8923 - loss: 0.0032
    Epoch 41/100
    438/438 - 1s - 1ms/step - accuracy: 0.8933 - loss: 0.0032
    Epoch 42/100
    438/438 - 1s - 1ms/step - accuracy: 0.8943 - loss: 0.0032
    Epoch 43/100
    438/438 - 1s - 1ms/step - accuracy: 0.8940 - loss: 0.0032
    Epoch 44/100
    438/438 - 1s - 1ms/step - accuracy: 0.8949 - loss: 0.0032
    Epoch 45/100
    438/438 - 1s - 1ms/step - accuracy: 0.8944 - loss: 0.0032
    Epoch 46/100
    438/438 - 1s - 1ms/step - accuracy: 0.8941 - loss: 0.0032
    Epoch 47/100
    438/438 - 1s - 1ms/step - accuracy: 0.8931 - loss: 0.0032
    Epoch 48/100
    438/438 - 1s - 1ms/step - accuracy: 0.8939 - loss: 0.0032
    Epoch 49/100
    438/438 - 1s - 1ms/step - accuracy: 0.8949 - loss: 0.0032
    Epoch 50/100
    438/438 - 1s - 1ms/step - accuracy: 0.8915 - loss: 0.0032
    Epoch 51/100
    438/438 - 1s - 1ms/step - accuracy: 0.8935 - loss: 0.0032
    Epoch 52/100
    438/438 - 1s - 1ms/step - accuracy: 0.8940 - loss: 0.0032
    Epoch 53/100
    438/438 - 1s - 1ms/step - accuracy: 0.8938 - loss: 0.0032
    Epoch 54/100
    438/438 - 1s - 1ms/step - accuracy: 0.8947 - loss: 0.0032
    Epoch 55/100
    438/438 - 1s - 1ms/step - accuracy: 0.8934 - loss: 0.0032
    Epoch 56/100
    438/438 - 1s - 1ms/step - accuracy: 0.8929 - loss: 0.0032
    Epoch 57/100
    438/438 - 1s - 1ms/step - accuracy: 0.8931 - loss: 0.0032
    Epoch 58/100
    438/438 - 1s - 1ms/step - accuracy: 0.8939 - loss: 0.0032
    Epoch 59/100
    438/438 - 1s - 1ms/step - accuracy: 0.8934 - loss: 0.0032
    Epoch 60/100
    438/438 - 1s - 1ms/step - accuracy: 0.8944 - loss: 0.0032
    Epoch 61/100
    438/438 - 1s - 1ms/step - accuracy: 0.8938 - loss: 0.0032
    Epoch 62/100
    438/438 - 1s - 1ms/step - accuracy: 0.8959 - loss: 0.0032
    Epoch 63/100
    438/438 - 1s - 1ms/step - accuracy: 0.8953 - loss: 0.0031
    Epoch 64/100
    438/438 - 1s - 1ms/step - accuracy: 0.8932 - loss: 0.0031
    Epoch 65/100
    438/438 - 1s - 1ms/step - accuracy: 0.8933 - loss: 0.0032
    Epoch 66/100
    438/438 - 1s - 1ms/step - accuracy: 0.8944 - loss: 0.0032
    Epoch 67/100
    438/438 - 1s - 1ms/step - accuracy: 0.8942 - loss: 0.0032
    Epoch 68/100
    438/438 - 1s - 1ms/step - accuracy: 0.8937 - loss: 0.0031
    Epoch 69/100
    438/438 - 1s - 1ms/step - accuracy: 0.8937 - loss: 0.0031
    Epoch 70/100
    438/438 - 1s - 1ms/step - accuracy: 0.8932 - loss: 0.0031
    Epoch 71/100
    438/438 - 1s - 1ms/step - accuracy: 0.8941 - loss: 0.0031
    Epoch 72/100
    438/438 - 1s - 1ms/step - accuracy: 0.8934 - loss: 0.0032
    Epoch 73/100
    438/438 - 1s - 1ms/step - accuracy: 0.8935 - loss: 0.0032
    Epoch 74/100
    438/438 - 1s - 1ms/step - accuracy: 0.8929 - loss: 0.0032
    Epoch 75/100
    438/438 - 1s - 1ms/step - accuracy: 0.8930 - loss: 0.0032
    Epoch 76/100
    438/438 - 1s - 1ms/step - accuracy: 0.8944 - loss: 0.0031
    Epoch 77/100
    438/438 - 1s - 1ms/step - accuracy: 0.8927 - loss: 0.0032
    Epoch 78/100
    438/438 - 1s - 1ms/step - accuracy: 0.8917 - loss: 0.0032
    Epoch 79/100
    438/438 - 1s - 1ms/step - accuracy: 0.8942 - loss: 0.0031
    Epoch 80/100
    438/438 - 1s - 1ms/step - accuracy: 0.8935 - loss: 0.0031
    Epoch 81/100
    438/438 - 1s - 1ms/step - accuracy: 0.8915 - loss: 0.0031
    Epoch 82/100
    438/438 - 1s - 1ms/step - accuracy: 0.8932 - loss: 0.0032
    Epoch 83/100
    438/438 - 1s - 1ms/step - accuracy: 0.8916 - loss: 0.0032
    Epoch 84/100
    438/438 - 1s - 1ms/step - accuracy: 0.8929 - loss: 0.0031
    Epoch 85/100
    438/438 - 1s - 1ms/step - accuracy: 0.8920 - loss: 0.0031
    Epoch 86/100
    438/438 - 1s - 1ms/step - accuracy: 0.8934 - loss: 0.0032
    Epoch 87/100
    438/438 - 1s - 1ms/step - accuracy: 0.8932 - loss: 0.0031
    Epoch 88/100
    438/438 - 1s - 1ms/step - accuracy: 0.8939 - loss: 0.0031
    Epoch 89/100
    438/438 - 1s - 1ms/step - accuracy: 0.8928 - loss: 0.0031
    Epoch 90/100
    438/438 - 1s - 1ms/step - accuracy: 0.8927 - loss: 0.0031
    Epoch 91/100
    438/438 - 1s - 1ms/step - accuracy: 0.8925 - loss: 0.0032
    Epoch 92/100
    438/438 - 1s - 1ms/step - accuracy: 0.8930 - loss: 0.0031
    Epoch 93/100
    438/438 - 1s - 1ms/step - accuracy: 0.8912 - loss: 0.0032
    Epoch 94/100
    438/438 - 1s - 1ms/step - accuracy: 0.8912 - loss: 0.0031
    Epoch 95/100
    438/438 - 1s - 1ms/step - accuracy: 0.8924 - loss: 0.0031
    Epoch 96/100
    438/438 - 1s - 1ms/step - accuracy: 0.8909 - loss: 0.0031
    Epoch 97/100
    438/438 - 1s - 1ms/step - accuracy: 0.8927 - loss: 0.0031
    Epoch 98/100
    438/438 - 1s - 1ms/step - accuracy: 0.8922 - loss: 0.0031
    Epoch 99/100
    438/438 - 1s - 1ms/step - accuracy: 0.8912 - loss: 0.0031
    Epoch 100/100
    438/438 - 1s - 1ms/step - accuracy: 0.8924 - loss: 0.0031
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.02551931 0.03266070 0.07641141 0.02703325 

![](CNN_SIR_all_files/figure-commonmark/give%20parameters%20and%20run-1.png)

![](CNN_SIR_all_files/figure-commonmark/give%20parameters%20and%20run-2.png)

# Section 2

Now we can Run a CNN model to find the best parameters we can use for
our CNN model to perform

This build_and_train_model function is designed to create, compile,
train, and evaluate a convolutional neural network (CNN) model using the
keras3 library in R for deep learning tasks.

``` r
build_and_train_model <- function(train, test, theta, seed,
                                  filters, kernel_size, activation_conv,
                                  activation_dense, pool_size, optimizer,
                                  loss, epochs, verbose = 0) {
  # Build the model
  model <- keras3::keras_model_sequential()
  model %>%
    keras3::layer_conv_2d(
      filters     = filters,
      input_shape = c(dim(train$x)[2], dim(train$x)[3], 1),
      activation  = activation_conv,
      kernel_size = kernel_size
    ) %>%
    keras3::layer_max_pooling_2d(
      pool_size = pool_size,
      padding = 'same'
    ) %>%
    keras3::layer_flatten() %>%
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
  model %>% keras3::fit(
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
```

- **Building the model**:

  - The function begins by creating a sequential CNN model using
    `keras3::keras_model_sequential()`.

  - The first layer is a 2D convolutional layer with customizable
    `filters`, `kernel_size`, and `activation_conv`, applied to input
    data of shape `(height, width, 1)` (grayscale images).

  - A max-pooling layer follows, which reduces spatial dimensions using
    a pool size (`pool_size`).

  - The model is then flattened to transition from 2D to 1D data.

  - A dense (fully connected) layer is added with the number of units
    equal to the number of columns in `theta` and an activation function
    `activation_dense`.

- **Compiling the model**:

  - The model is compiled with an optimizer (`optimizer`), loss function
    (`loss`), and the metric mean absolute error (`mae`) to track
    performance.

    # visualize results

``` r
#|label: Function to visualize results
visualize_results <- function(pred, test, theta, MAEs, N, N_train, output_file = NULL) {
  # Prepare the data for plotting
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
      sprintf("(MAE: %.2f)", MAEs)
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
        "Best Model with Mean MAE: %.2f",
        mean(MAEs)
      ),
      x = "Observed Values",
      y = "Predicted Values"
    )
  
  print(p2)
}
```

# hyperparameter tuning

The `main` function orchestrates a full machine learning workflow for
training a convolutional neural network (CNN) on simulated data.

``` r
main <- function(N, n, ndays, ncores) {
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
  # Optionally save the data
  # saveRDS(list(theta = theta2, simulations = arrays_1d), file = "calibration/sir.rds", compress = TRUE)
  
  # Split data into training and testing sets
  data_split <- split_data(arrays_1d, theta2, N)
  train <- data_split$train
  test <- data_split$test
  N_train <- data_split$N_train
  
  # Define hyperparameter grid
  hyper_grid <- expand.grid(
    filters = c(16, 32, 64),
    kernel_size = list(c(3,3), c(3,5)),
    activation_conv = c('relu', 'linear'),
    activation_dense = c('sigmoid'),
    pool_size = list(c(2,2)),
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
          theta = theta2,
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
  visualize_results(best_pred, test, theta2, best_MAEs, N, N_train)
}
```

### **1. Generate Parameters and Seeds**

- The function starts by generating a set of parameters (`theta`) using
  `generate_theta`, which creates a matrix based on `N` and `n`.

- It also generates random seeds (`seeds`) for each simulation using
  `sample.int()` to ensure reproducibility.

### 2. **Run Simulations**

- Simulations are run using `run_simulations`, which likely generates a
  set of matrices representing the simulated data. The inputs include
  the number of simulations (`N`), days (`ndays`), and the number of
  cores (`ncores`).

### 3. **Filter Non-Null Data**

- After simulations, non-null matrices are filtered out using
  `filter_non_null`, which removes invalid results and updates
  `matrices`, `theta`, and `N` accordingly.

### 4. **Prepare Data for TensorFlow**

- The simulation data is then transformed to be used in TensorFlow via
  `prepare_data_for_tensorflow`, which likely reshapes the data into the
  format needed for model training.

### 5. **Data Adjustment and Saving**

- The `theta` matrix is adjusted by converting the `crate` column using
  the logistic (sigmoid) function (`plogis(crate / 10)`).

- There is an option to save the data (`theta` and the reshaped arrays)
  using `saveRDS`, though it’s commented out.

### 6. **Split Data into Training and Testing Sets**

- The data is split into training and testing sets using `split_data`,
  which returns `train`, `test`, and the number of training examples
  (`N_train`).

### 7. **Define Hyperparameter Grid**

- A grid of hyperparameters is defined using `expand.grid()`, which
  includes options for:

  - Number of filters in the convolutional layer (`filters`).

  - Kernel sizes (`kernel_size`).

  - Activation functions for the convolutional and dense layers
    (`activation_conv`, `activation_dense`).

  - Pooling size (`pool_size`).

  - Optimizer (`adam`).

  - Loss functions (`mse`, `mae`).

  - Number of epochs (`epochs`).

### 8. **Hyperparameter Tuning Loop**

- A loop iterates through each combination of hyperparameters in
  `hyper_grid`.

- For each combination, the function prints the current model number and
  extracts the respective hyperparameters.

- A random seed (`seed = 331`) is set for reproducibility.

- The function then calls `build_and_train_model` to build, compile, and
  train the CNN model.

- If an error occurs during model training, it is caught using
  `tryCatch`, and the iteration moves to the next combination.

### 9. **Evaluate Models**

- After training, the model’s performance is evaluated using mean
  absolute errors (MAEs).

- If the current model’s average MAE is the best so far, the function
  stores the model, predictions, MAEs, and corresponding hyperparameters
  as the best model.

### 10. **Display Best Model**

- After the hyperparameter tuning loop, the function prints the best
  hyperparameters and the lowest MAE achieved.

### 11. **Visualize Results**

- Finally, the function calls `visualize_results` to display the
  predictions and compare them to the true values using the test set,
  `theta2`, and other relevant metrics.

# Results

``` r
# Run the main function with specified parameters
N <- 2e4   
n <- 5000
ndays <- 50
ncores <- 20
main(N, n, ndays, ncores)
```

    Testing model 1 of 24 
    188/188 - 0s - 1ms/step
    Testing model 2 of 24 
    188/188 - 0s - 1ms/step
    Testing model 3 of 24 
    188/188 - 0s - 1ms/step
    Testing model 4 of 24 
    188/188 - 0s - 1ms/step
    Testing model 5 of 24 
    188/188 - 0s - 1ms/step
    Testing model 6 of 24 
    188/188 - 0s - 1ms/step
    Testing model 7 of 24 
    188/188 - 0s - 1ms/step
    Testing model 8 of 24 
    188/188 - 0s - 1ms/step
    Testing model 9 of 24 
    188/188 - 0s - 1ms/step
    Testing model 10 of 24 
    188/188 - 0s - 1ms/step
    Testing model 11 of 24 
    188/188 - 0s - 1ms/step
    Testing model 12 of 24 
    188/188 - 0s - 1ms/step
    Testing model 13 of 24 
    188/188 - 0s - 1ms/step
    Testing model 14 of 24 
    188/188 - 0s - 1ms/step
    Testing model 15 of 24 
    188/188 - 0s - 1ms/step
    Testing model 16 of 24 
    188/188 - 0s - 1ms/step
    Testing model 17 of 24 
    188/188 - 0s - 1ms/step
    Testing model 18 of 24 
    188/188 - 0s - 1ms/step
    Testing model 19 of 24 
    188/188 - 0s - 1ms/step
    Testing model 20 of 24 
    188/188 - 0s - 1ms/step
    Testing model 21 of 24 
    188/188 - 0s - 1ms/step
    Testing model 22 of 24 
    188/188 - 0s - 1ms/step
    Testing model 23 of 24 
    188/188 - 0s - 1ms/step
    Testing model 24 of 24 
    188/188 - 0s - 1ms/step
    Best model parameters:
       filters kernel_size activation_conv activation_dense pool_size optimizer
    16      16        3, 5            relu          sigmoid      2, 2      adam
       loss epochs        MAE
    16  mae     50 0.03541653
    Best MAE: 0.03541653 

![](CNN_SIR_all_files/figure-commonmark/unnamed-chunk-18-1.png)

![](CNN_SIR_all_files/figure-commonmark/unnamed-chunk-18-2.png)
