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
    epochs = 10,
    verbose = 2
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

    Epoch 1/10
    438/438 - 1s - 3ms/step - accuracy: 0.6966 - loss: 0.0806
    Epoch 2/10
    438/438 - 1s - 1ms/step - accuracy: 0.6930 - loss: 0.0516
    Epoch 3/10
    438/438 - 1s - 1ms/step - accuracy: 0.7087 - loss: 0.0458
    Epoch 4/10
    438/438 - 1s - 1ms/step - accuracy: 0.7185 - loss: 0.0218
    Epoch 5/10
    438/438 - 1s - 1ms/step - accuracy: 0.7537 - loss: 0.0123
    Epoch 6/10
    438/438 - 1s - 1ms/step - accuracy: 0.7722 - loss: 0.0102
    Epoch 7/10
    438/438 - 1s - 1ms/step - accuracy: 0.7825 - loss: 0.0092
    Epoch 8/10
    438/438 - 1s - 1ms/step - accuracy: 0.7920 - loss: 0.0086
    Epoch 9/10
    438/438 - 1s - 1ms/step - accuracy: 0.7974 - loss: 0.0083
    Epoch 10/10
    438/438 - 1s - 1ms/step - accuracy: 0.8046 - loss: 0.0080
    188/188 - 0s - 1ms/step
        preval      crate      ptran       prec 
    0.04298311 0.06563423 0.08307995 0.10607962 

![](CNN_SIR_infected_only_files/figure-commonmark/Main%20execution%20function-2.png)

![](CNN_SIR_infected_only_files/figure-commonmark/Main%20execution%20function-3.png)

![](CNN_SIR_infected_only_files/figure-commonmark/Main%20execution%20function-4.png)