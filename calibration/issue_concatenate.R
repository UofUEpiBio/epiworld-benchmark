# Load necessary packages
library(epiworldR)
library(data.table)
library(tensorflow)
library(keras)
library(keras3)
library(parallel)
library(dplyr)
library(ggplot2)

# Set parameters
N <- 2e4
n <- 5000
ndays <- 50
ncores <- 20

# Function to generate parameters (theta) for simulations
generate_theta <- function(N, n) {
  set.seed(1231)
  theta <- data.table(
    preval = sample((100:2000) / n, N, TRUE),
    crate  = rgamma(N, shape = 5, rate = 1),  # Adjusted to use shape and rate parameter names
    ptran  = rbeta(N, 3, 7),   # Mean 0.3
    prec   = rbeta(N, 10, 10)  # Adjusted to correct logic, maintaining mean 0.5
  )
  return(theta)
}

# Function to run epidemic model simulations in parallel
run_simulations <- function(N, theta, seeds, n, ndays, ncores) {
  matrices <- parallel::mclapply(1:N, FUN = function(i) {
    fn <- sprintf("calibration/simulated_data/sir-%06i.rds", i)
    if (file.exists(fn)) return(readRDS(fn))  # Check if the file already exists
    set.seed(seeds[i])
    m <- ModelSIRCONN("mycon", prevalence = theta[i, preval], 
                      contact_rate = theta[i, crate], 
                      transmission_rate = theta[i, ptran], 
                      recovery_rate = theta[i, prec], 
                      n = n)
    verbose_off(m)
    run(m, ndays = ndays)
    ans <- prepare_data(m)
    saveRDS(ans, fn)
    return(ans)
  }, mc.cores = ncores)
  return(matrices)
}

# Filter valid simulations and update theta
filter_valid_simulations <- function(matrices, theta) {
  valid_indices <- intersect(
    which(!sapply(matrices, inherits, what = "error")),
    which(!sapply(matrices, function(x) any(is.na(x))))
  )
  matrices <- matrices[valid_indices]
  theta <- theta[valid_indices, ]
  return(list(matrices = matrices, theta = theta, N = length(valid_indices)))
}

# Prepare simulation data for neural network
prepare_neural_network_data <- function(matrices, N) {
  arrays_1d <- array(dim = c(N, dim(matrices[[1]][1,,])))
  for (i in seq_along(matrices)) {
    arrays_1d[i,,] <- matrices[[i]][1,,]
  }
  return(arrays_1d)
}

# Split train and test data for CNN and LSTM
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

# Build CNN model with correct input shape
build_cnn_model <- function(input_shape, output_units) {
  cnn_input <- layer_input(shape = input_shape, name = "CNN_Input")
  
  cnn_output <- cnn_input %>%
    layer_conv_2d(filters = 32, kernel_size = c(3, 5), activation = "relu") %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_flatten()%>% layer_dense(units = 64, activation = "relu")   # Flatten the output to make it compatible with LSTM output
  
  return(list(cnn_input = cnn_input, cnn_output = cnn_output))
}




# Build LSTM model
build_lstm_model <- function(input_shape, output_units) {
  lstm_input <- layer_input(shape = input_shape, name = "LSTM_Input")
  
  lstm_output <- lstm_input %>%
    layer_lstm(units = 64, return_sequences = FALSE) %>%
    layer_dense(units = 64, activation = "relu")
  
  return(list(lstm_input = lstm_input, lstm_output = lstm_output))
}

build_cnn_lstm_model <- function(cnn_input_shape, lstm_input_shape, output_units) {
  
  # Build CNN model
  cnn <- build_cnn_model(cnn_input_shape, output_units)
  
  # Build LSTM model
  lstm <- build_lstm_model(lstm_input_shape, output_units)
  
  # Flatten both CNN and LSTM outputs
  cnn_flattened <- cnn$cnn_output %>%
    layer_flatten()  
  lstm_flattened <- lstm$lstm_output 
  # # Already flattened after LSTM
  # 
  # print("Shape of CNN output after flattening:")
  # print(cnn_flattened$shape)
  # 
  # print("Shape of LSTM output:")
  # print(lstm_flattened$shape)
  # 
  
  concatenated=layer_concatenate(list(cnn_flattened,lstm_flattened),  axis = -1) 
  # Add dense layers for further processing after concatenation
  dense_layer <- concatenated %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 4, activation = "sigmoid")
  
  # Define the full model
  model <- keras::keras_model(
    inputs = concatenated,
    outputs = dense_layer
  )
  
  # Compile the model
  model %>% compile(optimizer = "adam", loss = "MAE", metrics = "accuracy")
  
  return(model)
}




# Train the model with both CNN and LSTM data
train_model <- function(model, train_data, epochs = 100, batch_size = 32) {
  set_random_seed(123)
  model %>% fit(
    x = train_data$x,
    y = train_data$y,
    epochs = epochs,
    batch_size = batch_size,
    verbose = 1
  )
}
library(tensorflow)
?tensorflow
library(keras)

# Evaluate the model
evaluate_model <- function(model, test_data) {
  results <- model %>% evaluate(
    x = (test_data$x),
    y = test_data$y
  )
  return(results)
}


##Debugging:

main_pipeline <- function(N, n, ndays, ncores) {
  # Step 1: Generate theta
  theta <- generate_theta(N, n)
  
  
  # Print the shape of theta
  print("Shape of theta:")
  print(dim(theta))
  
  # Step 2: Generate random seeds
  seeds <- sample.int(.Machine$integer.max, N, TRUE)
  
  # Step 3: Run epidemic simulations
  matrices <- run_simulations(N, theta, seeds, n, ndays, ncores)
  
  # Step 4: Filter valid simulations
  valid_data <- filter_valid_simulations(matrices, theta)
  matrices <- valid_data$matrices
  theta <- valid_data$theta
  N <- valid_data$N
  
  # Print the number of valid simulations
  print(paste("Number of valid simulations:", N))
  
  # Step 5: Prepare data for the neural network
  nn_data <- prepare_neural_network_data(matrices, N)
  arrays_1d <- nn_data
  
  # Print the shape of the prepared data for the neural network
  print("Shape of arrays_1d (input data for NN):")
  print(dim(arrays_1d))
  
  # Step 6: Split train and test data
  data_split <- split_data(arrays_1d, theta, N)
  train <- data_split$train
  test <- data_split$test
  
  # Print shapes of training and testing data
  print("Shape of train$x:")
  print(dim(train$x))
  print("Shape of train$y:")
  print(dim(train$y))
  
  print("Shape of test$x:")
  print(dim(test$x))
  print("Shape of test$y:")
  print(dim(test$y))
  
  # Step 7: Build and train the CNN-LSTM model
  print("Building CNN and LSTM models separately...")
  
  # Debugging CNN and LSTM input shapes
  cnn_input_shape <- c(dim(train$x)[2], dim(train$x)[3], 1)  # 3D for CNN
  lstm_input_shape <- c(dim(train$x)[2], dim(train$x)[3])    # 2D for LSTM
  
  print("CNN input shape:")
  print(cnn_input_shape)
  
  print("LSTM input shape:")
  print(lstm_input_shape)
  
  # Build the CNN model
  cnn <- build_cnn_model(cnn_input_shape, output_units = ncol(theta))
  
  # Build the LSTM model
  lstm <- build_lstm_model(lstm_input_shape, output_units = ncol(theta))
  
  # Print the CNN model summary separately
  print("CNN Model Summary:")
  cnn_model <- keras_model(inputs = cnn$cnn_input, outputs = cnn$cnn_output)
  summary(cnn_model)  # This prints the CNN model structure
  
  # Print the LSTM model summary separately
  print("LSTM Model Summary:")
  lstm_model <- keras_model(inputs = lstm$lstm_input, outputs = lstm$lstm_output)
  summary(lstm_model)  # This prints the LSTM model structure
  
  # Step 8: Build and train the combined CNN-LSTM model
  print("Building the combined CNN-LSTM model...")
  
  model <- build_cnn_lstm_model(
    cnn_input_shape = cnn_input_shape, 
    lstm_input_shape = lstm_input_shape, 
    output_units = ncol(theta)
  )
  
  # Print the combined model summary
  print("Combined CNN-LSTM Model Summary:")
  summary(model)
  
  # Step 9: Train the model
  print("Training the combined model...")
  train_model(model, train, epochs = 100)
  
  # Step 10: Evaluate the model
  print("Evaluating the combined model...")
  eval_results <- evaluate_model(model, test)
  
  # Print evaluation results
  print("Evaluation Results:")
  print(eval_results)
}

















# Main execution block
main_pipeline <- function(N, n, ndays, ncores) {
  # Step 1: Generate theta
  theta <- generate_theta(N, n)
  
  # Step 2: Generate random seeds
  seeds <- sample.int(.Machine$integer.max, N, TRUE)
  
  # Step 3: Run epidemic simulations
  matrices <- run_simulations(N, theta, seeds, n, ndays, ncores)
  
  # Step 4: Filter valid simulations
  valid_data <- filter_valid_simulations(matrices, theta)
  matrices <- valid_data$matrices
  theta <- valid_data$theta
  N <- valid_data$N
  
  # Step 5: Prepare data for the neural network
  nn_data <- prepare_neural_network_data(matrices, N)
  arrays_1d <- nn_data
  
  # Step 6: Split train and test data
  data_split <- split_data(arrays_1d, theta, N)
  train <- data_split$train
  test <- data_split$test
  
  # Step 7: Build and train the CNN-LSTM model
  model <- build_cnn_lstm_model(
    cnn_input_shape = c(dim(train$x)[2], dim(train$x)[3], 1),  # 3D for CNN
    lstm_input_shape = c(dim(train$x)[2], dim(train$x)[3]),    # 2D for LSTM
    output_units = ncol(theta)
  )
  
  # Step 8: Train the model
  train_model(model, train, epochs = 100)
  
  # Step 9: Evaluate the model
  eval_results <- evaluate_model(model, test)
  print(eval_results)
}

# Execute the pipeline
main_pipeline(N, n, ndays, ncores)





