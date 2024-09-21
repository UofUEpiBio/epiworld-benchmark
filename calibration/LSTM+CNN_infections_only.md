# Implementing the CNN+LSTM Model by Knowing infected Counts to Find the
Closest Parameters to Simulate SIR Models

Installing Packages if necessary:

``` r
# Load Required Libraries
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

# Simulate Data and Theta

This function simulates data based on generating theta which is our
parameters.

``` r
#|label: Simulate Data
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
    
    verbose_off(m)
    run(m, ndays = ndays)
    ans <- prepare_data(m)
    saveRDS(ans, fn)
    ans
  }, mc.cores = ncores)
  
  is_not_null <- intersect(
    which(!sapply(matrices, inherits, what = "error")),
    which(!sapply(matrices, function(x) any(is.na(x))))
  )
  
  matrices <- matrices[is_not_null]
  theta    <- theta[is_not_null,]
  
  N <- length(is_not_null)
  
  arrays_1d <- array(dim = c(N, dim(matrices[[1]][1,,])))
  for (i in seq_along(matrices))
    arrays_1d[i,,] <- matrices[[i]][1,,]
  
  theta2 <- copy(theta)
  theta2[, crate := plogis(crate / 10)]
  
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

# Prepare Data for CNN and LSTM

This Function prepares data for TensorFlow and then splits it to train
and test. Also, it just uses the counts of infections in the dataset.

``` r
prepare_data_sets <- function(datafile = "calibration/sir.rds", train_fraction = 0.7) {
  sim_results <- readRDS(datafile)
  theta <- sim_results$theta
  arrays_1d <- sim_results$simulations
  
  arrays_1d <- arrays_1d[,1,,drop=FALSE]
  N <- dim(arrays_1d)[1]
  
  N_train <- floor(N * train_fraction)
  id_train <- 1:N_train
  train <- list(
    x = array_reshape(arrays_1d[id_train,,], dim = c(N_train, dim(arrays_1d)[-1])),
    y = as.matrix(theta)[id_train,]
  )
  
  N_test <- N - N_train
  id_test <- (N_train + 1):N
  test <- list(
    x = array_reshape(arrays_1d[id_test,,], dim = c(N_test, dim(arrays_1d)[-1])),
    y = as.matrix(theta)[id_test,]
  )
  
  list(train = train, test = test, theta = theta, arrays_1d = arrays_1d, N = N, N_train = N_train, N_test = N_test)
}
```

# CNN Model

``` r
build_and_train_CNN <- function(train, test, arrays_1d, theta, N_train, seed = 331, save_model_file = "sir-cnn_infections_only") {
  model <- keras3::keras_model_sequential()
  model |>
    keras3::layer_conv_2d(
      filters     = 32,
      input_shape = c(dim(arrays_1d)[-1], 1),
      activation  = "relu",
      kernel_size = c(1, 5)
    ) |>
    keras3::layer_max_pooling_2d(pool_size = 2, padding = 'same') |>
    keras3::layer_flatten() |>
    keras3::layer_dense(
      units = ncol(theta),
      activation = 'sigmoid'
    )
  
  model %>% compile(
    optimizer = 'adam',
    loss      = 'mse',
    metric    = 'mae'
  )
  
  tensorflow::set_random_seed(seed)
  model |> fit(train$x, train$y, epochs = 50, verbose = 0)
  
  pred <- predict(model, x = test$x) |> as.data.table() |> setnames(colnames(theta))
  MAEs <- abs(pred - as.matrix(test$y)) |> colMeans() |> print()
  
  list(pred = pred, MAEs = MAEs)
}
```

# LSTM Model

``` r
build_and_train_LSTM <- function(train, test, theta, N_train, ndays, seed = 331, save_model_file = "sir-lstm_infections_only") {
  model <- keras_model_sequential() %>%
    layer_lstm(units = 50, input_shape = c(ndays, 1)) %>%
    layer_dense(units = ncol(theta), activation = 'sigmoid')
  
  model %>% compile(optimizer = 'adam', loss = 'mse', metrics = 'mae')
  
  tensorflow::set_random_seed(seed)
  history <- model %>% fit(x = train$x, y = train$y, epochs = 50, batch_size = 64, validation_split = 0.2, verbose = 0)
  
  pred <- predict(model, x = test$x) %>%
    as.data.table() %>%
    setnames(colnames(theta))
  
  MAEs <- abs(pred - as.matrix(test$y)) %>%
    colMeans() %>%
    print()
  
  list(pred = pred, MAEs = MAEs, history = history)
}
```

# Ensemble Prediction (Simple and Weighted)

The `ensemble_predictions` function combines the predictions from two
models—CNN (Convolutional Neural Network) and LSTM (Long Short-Term
Memory) networks. This is done in two ways: simple averaging and
weighted averaging.

``` r
ensemble_predictions <- function(cnn_pred, lstm_pred, weight_cnn = 0.3, weight_lstm = 0.7) {
  # Simple average
  ensemble_avg <- (cnn_pred + lstm_pred) / 2
  
  # Weighted average
  ensemble_weighted <- (cnn_pred * weight_cnn) + (lstm_pred * weight_lstm)
  
  list(average = ensemble_avg, weighted = ensemble_weighted)
}

# MAE Calculation for Ensemble
calculate_MAE <- function(pred, actual) {
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
  ndays <- 50
  
  # Reshape data for CNN and LSTM
  train_cnn <- train
  test_cnn <- test
  train_lstm <- list(x = aperm(train$x, c(1, 3, 2)), y = train$y)
  test_lstm <- list(x = aperm(test$x, c(1, 3, 2)), y = test$y)
  
  # Build and train the CNN model
  cnn_results <- build_and_train_CNN(train_cnn, test_cnn, arrays_1d, theta, N_train)
  cnn_pred <- cnn_results$pred
  cnn_MAEs <- cnn_results$MAEs
  
  # Build and train the LSTM model
  lstm_results <- build_and_train_LSTM(train_lstm, test_lstm, theta, N_train, ndays)
  lstm_pred <- lstm_results$pred
  lstm_MAEs <- lstm_results$MAEs
  
  # Ensemble predictions
  ensemble_results <- ensemble_predictions(cnn_pred, lstm_pred)
  ensemble_avg <- ensemble_results$average
  ensemble_weighted <- ensemble_results$weighted
  
  # Calculate MAEs for ensemble
  avg_MAE <- calculate_MAE(ensemble_avg, test$y)
  weighted_MAE <- calculate_MAE(ensemble_weighted, test$y)
  
  # Visualize combined results
  visualize_combined_results(cnn_pred, lstm_pred, ensemble_avg, ensemble_weighted, test, theta, cnn_MAEs, lstm_MAEs, avg_MAE, weighted_MAE, N, N_train)
}

# Run the main function
main()
```


    Attaching package: 'epiworldR'

    The following object is masked from 'package:keras3':

        clone_model

    The following object is masked from 'package:keras':

        clone_model

    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.03336936 0.03556279 0.07727944 0.07637288 
    188/188 - 1s - 4ms/step
        preval      crate      ptran       prec 
    0.02134892 0.03188877 0.07128318 0.05669068 
        preval      crate      ptran       prec 
    0.02413016 0.03300168 0.07309920 0.06329643 
        preval      crate      ptran       prec 
    0.02197034 0.03235203 0.07204308 0.05968686 

![](LSTM+CNN_infections_only_files/figure-commonmark/Main%20Execution%20for%20CNN,%20LSTM,%20and%20Combining%20Results-1.png)

![](LSTM+CNN_infections_only_files/figure-commonmark/Main%20Execution%20for%20CNN,%20LSTM,%20and%20Combining%20Results-2.png)

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

    - This helps visually assess each model’s performance and ensemble
      methods.
