# Implementing the CNN+LSTM Model by Knowing SIR Counts to Find the
Closest Parameters to Simulate SIR Models

Installing Packages if necessary:

``` r
#|label: Load Required Libraries
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

# generate theta parameters

``` r
#|label: Function to generate theta parameters
generate_theta <- function(N, n) {
  set.seed(1231)
  theta <- data.table(
    preval = sample((100:2000) / n, N, TRUE),
    crate  = rgamma(N, 5, 1),    # Mean 10
    ptran  = rbeta(N, 3, 7),     # Mean 3/(3 + 7) = 0.3
    prec   = rbeta(N, 10, 10 * 2 - 10) # Mean 10 / (10 * 2 - 10) = 0.5
  )
  return(theta)
}
```

# run simulations

``` r
#|label: Function to run simulations
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
```

``` r
#|label: Function to filter non-null matrices and update theta
filter_non_null <- function(matrices, theta) {
  is_not_null <- intersect(
    which(!sapply(matrices, inherits, what = "error")),
    which(!sapply(matrices, function(x) any(is.na(x))))
  )
  
  matrices <- matrices[is_not_null]
  theta    <- theta[is_not_null, ]
  
  return(list(matrices = matrices, theta = theta, N = length(is_not_null)))
}
```

# Prepare Data for CNN and LSTM

``` r
prepare_data_sets <- function(matrices, N) {
  arrays_1d <- array(dim = c(N, dim(matrices[[1]][1,,])))
  
  for (i in seq_along(matrices)) {
    arrays_1d[i,,] <- matrices[[i]][1,,]
  }
  
  return(arrays_1d)
}
```

# Split Data into Training and Testing Sets

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

# CNN Model

``` r
#|label: CNN Model
build_cnn_model <- function(input_shape, output_units) {
  model <- keras3::keras_model_sequential() %>%
    keras3::layer_conv_2d(
      filters = 32,
      input_shape = c(input_shape, 1),
      activation = "linear",
      kernel_size = c(3, 5)
    ) %>%
    keras3::layer_max_pooling_2d(pool_size = 2, padding = 'same') %>%
    keras3::layer_flatten() %>%
    keras3::layer_dense(units = output_units, activation = 'sigmoid')
  
  model %>% compile(optimizer = 'adam', loss = 'mse', metrics = 'accuracy')
  return(model)
}

# Train CNN Model
train_cnn_model <- function(model, train_data, epochs = 50) {
  tensorflow::set_random_seed(331)
  model %>% fit(train_data$x, train_data$y, epochs = epochs, verbose = 0)
}
```

# LSTM Model

``` r
build_lstm_model <- function(input_shape, output_units) {
  model <- keras_model_sequential() %>%
    layer_lstm(units = 64, input_shape = input_shape) %>%
    layer_dense(units = output_units, activation = 'sigmoid')
  
  model %>% compile(optimizer = 'adam', loss = 'mse', metrics = 'mae')
  return(model)
}

# Train LSTM Model
train_lstm_model <- function(model, train_data, epochs = 50) {
  tensorflow::set_random_seed(331)
  model %>% fit(train_data$x, train_data$y, epochs = epochs, verbose = 0)
}

# Evaluation Function for both models
evaluate_model <- function(model, test_data, theta) {
  pred <- predict(model, x = test_data$x) %>%
    as.data.table() %>%
    setnames(colnames(theta))
  
  MAEs <- abs(pred - as.matrix(test_data$y)) %>%
    colMeans() %>%
    print()
  
  return(list(pred = pred, MAEs = MAEs))
}
```

# Ensemble Prediction (Simple and Weighted)

The `ensemble_predictions` function combines the predictions from two
models—CNN (Convolutional Neural Network) and LSTM (Long Short-Term
Memory) networks. This is done in two ways: simple averaging and
weighted averaging.

``` r
# Ensemble Prediction (Simple and Weighted)
ensemble_predictions <- function(cnn_pred, lstm_pred, weight_cnn = 0.3, weight_lstm = 0.7) {
  avg_pred <- (cnn_pred + lstm_pred) / 2
  weighted_pred <- (cnn_pred * weight_cnn) + (lstm_pred * weight_lstm)
  
  return(list(average = avg_pred, weighted = weighted_pred))
}

# MAE Calculation for Ensemble
calculate_mae <- function(pred, actual) {
  mae <- abs(pred - actual) %>%
    colMeans() %>%
    print()
  return(mae)
}
```

#### **Function Inputs:**

- **`cnn_pred`**: Predictions from the CNN model.

- **`lstm_pred`**: Predictions from the LSTM model.

- **`weight_cnn`**: The weight given to the CNN model in the weighted
  average (default is 0.3).

- **`weight_lstm`**: The weight given to the LSTM model in the weighted
  average (default is 0.7).

#### **Steps Inside the Function:**

1.  **Simple Average Ensemble**:

    - `avg_pred <- (cnn_pred + lstm_pred) / 2`: This line computes the
      simple average of the predictions from the CNN and LSTM models.
      Each element in the CNN and LSTM prediction arrays is averaged,
      assuming equal importance for both models.

2.  **Weighted Average Ensemble**:

    - `weighted_pred <- (cnn_pred * weight_cnn) + (lstm_pred * weight_lstm)`:

      - Here, the function computes a weighted average of the CNN and
        LSTM predictions.

      - **`weight_cnn`** (default 0.3) and **`weight_lstm`** (default
        0.7) specify the importance of each model’s predictions.

      - In this case, the LSTM predictions are given more weight (70%)
        compared to the CNN predictions (30%), which assumes the LSTM
        model performs better.

# Visualization Function for Results

The function `visualize_combined_results` provides a way to visualize
and compare predictions from CNN, LSTM, and ensemble models against the
actual (observed) values. This function creates two types of plots: a
**boxplot** and a **scatter plot**.

``` r
#|label: Visualization Function for Results
visualize_combined_results <- function(cnn_pred, lstm_pred, ensemble_avg, ensemble_weighted, test, theta, cnn_MAEs, lstm_MAEs, avg_MAE, weighted_MAE, N, N_train) {
  cnn_pred[, id := 1L:.N]
  lstm_pred[, id := 1L:.N]
  ensemble_avg[, id := 1L:.N]
  ensemble_weighted[, id := 1L:.N]
  
  theta_long <- as.data.table(test$y)
  setnames(theta_long, names(theta))
  theta_long[, id := 1L:.N]
  
  cnn_long <- melt(cnn_pred, id.vars = "id")
  lstm_long <- melt(lstm_pred, id.vars = "id")
  avg_long <- melt(ensemble_avg, id.vars = "id")
  weighted_long <- melt(ensemble_weighted, id.vars = "id")
  theta_long <- melt(theta_long, id.vars = "id")
  
  alldat <- rbind(
    cbind(cnn_long, Model = "CNN"),
    cbind(lstm_long, Model = "LSTM"),
    cbind(avg_long, Model = "Ensemble Avg"),
    cbind(weighted_long, Model = "Ensemble Weighted"),
    cbind(theta_long, Model = "Observed")
  )
  
  # Boxplot
  p1 <- ggplot(alldat, aes(x = value, colour = Model)) +
    facet_wrap(~variable, scales = "free") +
    geom_boxplot() +
    labs(title = "Boxplots of CNN, LSTM, and Ensemble Predictions with Observed Values")
  
  print(p1)
  
  # Scatter Plot
  alldat_wide <- dcast(alldat, id + variable ~ Model, value.var = "value")
  vnames <- data.table(
    variable = c("preval", "crate", "ptran", "prec"),
    Name     = paste(
      c("Init. state", "Contact Rate", "P(transmit)", "P(recover)"),
      sprintf("(CNN MAE: %.2f, LSTM MAE: %.2f, Avg MAE: %.2f, Weighted MAE: %.2f)", cnn_MAEs, lstm_MAEs, avg_MAE, weighted_MAE)
    )
  )
  
  alldat_wide <- merge(alldat_wide, vnames, by = "variable")
  
  p2 <- ggplot(alldat_wide, aes(x = Observed, y = CNN, colour = "CNN")) +
    geom_point(alpha = .2) +
    geom_point(aes(x = Observed, y = LSTM, colour = "LSTM"), alpha = .2) +
    geom_point(aes(x = Observed, y = `Ensemble Avg`, colour = "Ensemble Avg"), alpha = .2) +
    geom_point(aes(x = Observed, y = `Ensemble Weighted`, colour = "Ensemble Weighted"), alpha = .2) +
    facet_wrap(~ Name, scales = "free") +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
    labs(title = "Observed vs CNN, LSTM, and Ensemble Predictions (Validation Set)")
  
  print(p2)
}
```

### **Inputs:**

1.  **`cnn_pred`**: Predictions from the CNN model.

2.  **`lstm_pred`**: Predictions from the LSTM model.

3.  **`ensemble_avg`**: Predictions from the simple average of CNN and
    LSTM.

4.  **`ensemble_weighted`**: Predictions from the weighted average of
    CNN and LSTM.

5.  **`test`**: Test data containing the actual (observed) values.

6.  **`theta`**: A data table representing the parameters used for the
    simulations (like `preval`, `crate`, etc.).

7.  **`cnn_MAEs`**: Mean Absolute Errors for the CNN model predictions.

8.  **`lstm_MAEs`**: Mean Absolute Errors for the LSTM model
    predictions.

9.  **`avg_MAE`**: Mean Absolute Error for the simple average ensemble
    predictions.

10. **`weighted_MAE`**: Mean Absolute Error for the weighted average
    ensemble predictions.

11. **`N`**: Total number of data points (observations).

12. **`N_train`**: Number of training samples.

# Main Execution for CNN, LSTM, and Combining Results

The provided `main` function orchestrates the workflow for running a CNN
and LSTM model, evaluating their predictions, combining them using
ensemble methods, and visualizing the results.

``` r
#|label: Main Execution for CNN, LSTM, and Combining Results
main <- function(N = 2e4, n = 5000, ndays = 50, ncores = 20) {
  # Generate theta and seeds
  theta <- generate_theta(N, n)
  seeds <- sample.int(.Machine$integer.max, N, TRUE)
  
  # Run simulations
  matrices <- run_simulations(N, n, ndays, ncores, theta, seeds)
  
  # Filter non-null data
  filtered_data <- filter_non_null(matrices, theta)
  matrices <- filtered_data$matrices
  theta <- filtered_data$theta
  N <- filtered_data$N
  
  # Prepare Data
  arrays_1d <- prepare_data_sets(matrices, N)
  theta2 <- copy(theta)
  theta2[, crate := plogis(crate / 10)]
  
  # Split Data into Training and Testing Sets
  data_split <- split_data(arrays_1d, theta2, N)
  train <- data_split$train
  test <- data_split$test
  N_train <- data_split$N_train
  
  # CNN Model
  cnn_model <- build_cnn_model(input_shape = dim(arrays_1d)[-1], output_units = ncol(theta2))
  train_cnn_model(cnn_model, train)
  cnn_results <- evaluate_model(cnn_model, test, theta2)
  cnn_pred <- cnn_results$pred
  cnn_MAEs <- cnn_results$MAEs
  
  # LSTM Model
  lstm_train_data <- list(x = aperm(train$x, c(1, 3, 2)), y = train$y)
  lstm_test_data <- list(x = aperm(test$x, c(1, 3, 2)), y = test$y)
  lstm_model <- build_lstm_model(input_shape = c(dim(lstm_train_data$x)[2:3]), output_units = ncol(theta2))
  train_lstm_model(lstm_model, lstm_train_data)
  lstm_results <- evaluate_model(lstm_model, lstm_test_data, theta2)
  lstm_pred <- lstm_results$pred
  lstm_MAEs <- lstm_results$MAEs
  
  # Ensemble Predictions
  ensemble_results <- ensemble_predictions(cnn_pred, lstm_pred)
  avg_pred <- ensemble_results$average
  weighted_pred <- ensemble_results$weighted
  
  # Calculate MAEs for Ensemble
  avg_MAE <- calculate_mae(avg_pred, test$y)
  weighted_MAE <- calculate_mae(weighted_pred, test$y)
  
  # Visualize Results
  visualize_combined_results(cnn_pred, lstm_pred, avg_pred, weighted_pred, test, theta2, cnn_MAEs, lstm_MAEs, avg_MAE, weighted_MAE, N, N_train)
}

# Run the main function
main()
```

    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.01416299 0.03299895 0.07781534 0.02450230 
    188/188 - 1s - 5ms/step
        preval      crate      ptran       prec 
    0.01155348 0.03165723 0.07153182 0.01199014 
         preval       crate       ptran        prec 
    0.008715704 0.031880243 0.073099728 0.014254188 
         preval       crate       ptran        prec 
    0.008880406 0.031690307 0.072101659 0.011736138 

![](CNN+LSTM_all_files/figure-commonmark/unnamed-chunk-19-1.png)

![](CNN+LSTM_all_files/figure-commonmark/unnamed-chunk-19-2.png)

### **Explanation of the Workflow:**

1.  **Generate Parameters (`theta`) and Seeds:**

    - The function `generate_theta` creates the initial parameters for
      the simulation, which include the prevalence, contact rate,
      transmission rate, and recovery rate.

    - Random seeds are generated to ensure that the simulations can be
      replicated.

2.  **Run Simulations:**

    - The function `run_simulations` is used to generate simulated SIR
      data using the parameters (`theta`) and seeds. These simulations
      provide a time-series dataset for each of the `N` simulations.

    - The results are stored in matrices that contain the simulated SIR
      data.

3.  **Filter and Prepare Data:**

    - The function `filter_non_null` filters out simulations that
      generated invalid results or contain missing values.

    - The valid matrices are then reshaped using the `prepare_data_sets`
      function, which converts them into a format suitable for neural
      network training.

4.  **Split Data into Training and Testing Sets:**

    - The function `split_data` splits the prepared data into training
      and testing sets.

    - This is done to allow the models to be trained on a portion of the
      data and tested on unseen data to evaluate their performance.

5.  **CNN Model:**

    - The CNN model is built using `build_cnn_model`. It is a
      convolutional neural network that learns patterns from the input
      data (time-series).

    - The CNN model is trained using the `train_cnn_model` function, and
      its predictions are evaluated using the `evaluate_model` function.

    - The resulting predictions (`cnn_pred`) and Mean Absolute Errors
      (MAEs) (`cnn_MAEs`) are saved.

6.  **LSTM Model:**

    - The LSTM model is built using `build_lstm_model`. It is a Long
      Short-Term Memory network designed for sequential data, making it
      suitable for time-series prediction.

    - The training and testing data for the LSTM are reshaped
      accordingly.

    - The LSTM model is trained with `train_lstm_model`, and its
      performance is evaluated using the `evaluate_model` function.

    - The resulting predictions (`lstm_pred`) and MAEs (`lstm_MAEs`) are
      saved.

7.  **Ensemble Predictions:**

    - The predictions from both the CNN and LSTM models are combined
      using two ensemble methods in the `ensemble_predictions` function:

      - **Simple Average**: Takes the average of the CNN and LSTM
        predictions.

      - **Weighted Average**: Combines the predictions with a weighted
        average, where you can adjust the contribution from each model
        using the weights `weight_cnn` and `weight_lstm`.

    - These ensemble predictions are stored as `avg_pred` (simple
      average) and `weighted_pred` (weighted average).

8.  **Calculate MAEs for Ensemble:**

    - The function `calculate_mae` computes the Mean Absolute Error for
      both the simple and weighted ensemble predictions. These MAEs
      (`avg_MAE` and `weighted_MAE`) provide insights into how well the
      ensemble methods performed compared to the individual models.

9.  **Visualization:**

    - The function `visualize_combined_results` is used to create plots
      that compare the predictions from the CNN, LSTM, ensemble models,
      and the actual observed values.

    - The visualizations include:

      - **Boxplots**: Showing the spread of predictions and actual
        values.

      - **Scatter Plots**: Comparing observed vs. predicted values with
        different models.

    - This helps in visually assessing the performance of each model and
      the ensemble methods.
