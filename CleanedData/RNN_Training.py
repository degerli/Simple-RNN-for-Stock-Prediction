''' The RNN will have 3 layers, 50 recurrent neurons in that layer, and the neurons will 
be unraveled over 10 time steps since each training instance will be 10 inputs long. Each 
training instance is a randomly selected sequence of 10 consecutive days. The 
targets are a sequence of inputs, but shifted 1 day into the future.'''

import os
import numpy as np, pandas as pd, tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # To prevent tensorflow from printing out INFO messages

def next_batch (batch_size, n_steps, df):
    ''' 
    This function takes in a batch size, number of time steps, and a dataframe. 
    It returns the train (x) and target (y) values of a certain batch as
    arrays (size: batch_size, n_steps, 1). x is composed of randomly selected sequences from
    the dataframe, and y is the x sequence, but shifted over once. 
    The columns in x, y are the sequences, and the rows are instances.
    '''
    final_x = np.empty((1, n_steps, df.shape[1]))
    final_y = np.empty((1, n_steps, df.shape[1]))
    # Find a sequence, then concatenate it vertically.
    for i in range (batch_size):
        # Get a random number between 0 and the maximum row of dataframe - number of steps - 1 (to account for shifting of 1 in the future)
        random_number = np.random.randint(0, df.shape[0] - n_steps - 1)
        x = df[random_number: random_number + n_steps].values
        x = np.reshape(x, (1, n_steps, df.shape[1]))
        y = df[random_number + 1: random_number + n_steps + 1].values
        y = np.reshape(y, (1, n_steps, df.shape[1]))
        final_x = np.vstack((final_x, x)) 
        final_y = np.vstack((final_y, y))
    return final_x[1:], final_y[1:] # Index except the first to ignore the placeholder of 0s.


''' Leaky RELU is here if we want to switch or if the computing time is too slow'''
def leaky_relu (z, name=None):
    return tf.maximum(.01*z, z, name=name)

def create_RNN_layers (train, n_steps, columns, n_neurons, n_layers, learning_rate, n_iterations, batch_size, train_keep_prob, file_name):
    ''' 
    This function takes in the train set, number of unraveled time steps, the stock columns (OHLC, volume), number of neurons in each layer, number of layers, learning rate, number of iterations for training, batch size of the training, and the drop out probability. 
    '''

    n_inputs = len (columns)
    n_outputs = len (columns)
    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
    keep_prob = tf.placeholder_with_default(1.0, shape=()) # For Dropout
    
    '''
    Creates the layers and introduces dropout to reduce overfitting.
    GRUCell creates copies of the cell to build the unrolled RNN (one for each time step). GRU over LSTM because it is faster.
    We will use ELU for faster computing time and to avoid the dying ReLU problems.

    For each time step, the neurons output a vector of size equal to the number of neurons. But we want a single output at each time step, so we use OutputProjectionWrapper.
    '''
    layers = [tf.contrib.rnn.GRUCell(num_units=n_neurons,activation=tf.nn.elu)
             for layer in range(n_layers)]
    layers_drop = [tf.contrib.rnn.DropoutWrapper (layer, input_keep_prob = keep_prob)
                  for layer in layers]

    # Suggestion: introduce batch normalization.
    multi_layer_cells = tf.contrib.rnn.OutputProjectionWrapper(
        tf.contrib.rnn.MultiRNNCell(layers_drop), 
        output_size=n_outputs)
    
    # dynamic_rnn runs through the cell a number of times for each instance and for each time step.
    outputs, states = tf.nn.dynamic_rnn(multi_layer_cells, X, dtype=tf.float32)
    
    # Training
    loss = tf.reduce_mean(tf.square(outputs - y)) # MSE
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) # Best optimizer
    training_op = optimizer.minimize(loss)
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        init.run()
        for iteration in range(n_iterations):
            X_batch, y_batch = next_batch(batch_size, n_steps, train[columns])
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, keep_prob: train_keep_prob})
            if iteration % 200 == 0:
                mse = loss.eval(feed_dict={X: X_batch, y: y_batch, keep_prob: train_keep_prob})
                print(iteration, "\tMSE:", mse)

        saver.save(sess, "checkpoint/" + file_name + 'ts_model')
        
    return outputs, states, saver, X, y 

def predict_next_days (test, plot_column, n_steps, outputs, n_future_iter, file_name, saver, X, y):
    ''' 
    This function takes in the test data, the single column used to predicting stocks (like the close price), number of unraveled time steps, outputs of dynamic rnn, number of future iterations that one wants to predict, the file name of the restored tensorflow graph, X placeholder, and y placeholder.
    It restores the checkpoint created by create_RNN_layers, and uses the first n_steps to create the prediction of one time step shifted over (but we will take the last instance)
    The output is the first n_steps of the stock values used to predict, then the predicted values (there are n_future_iter of them).
    '''
    columns = test.columns
    n_inputs = len (columns)
    n_test_instances = test.shape[0]

    with tf.Session() as sess:                        
        saver.restore(sess, "checkpoint/" + file_name + 'ts_model')

        input_data = test[columns].values 
        sequence = test[plot_column][:n_steps].tolist() # starting sequence
        #print (sequence)
        for iteration in range(n_future_iter):
            X_batch = input_data[:n_steps].reshape(-1, n_steps, n_inputs)
            y_pred = sess.run(outputs, feed_dict={X: X_batch})
            #print (y_pred, y_pred[0, -1, 1])
            sequence.append(y_pred[0, -1, 3]) # 3 is the position of the close
            input_data = np.vstack((input_data, y_pred[0, -1]))
        output_pred = np.asarray(sequence)
    return output_pred

def create_predictions_for_entire_test_set (test, plot_column, n_steps, outputs, n_future_iter, file_name, saver, X, y):
    '''
    This function creates prediction for the entire test set. 
    It takes in the same variables as predict_next_days.
    The function predicts the next days (based on n_future_iter), appends it to the starting sequence, then drops the first instance of the test_copy to update.
    The function returns an array of the prediction.
    '''
    test_copy = test.copy()
    n_test_rows = test.shape[0]
    sequence = test[plot_column][:n_steps].values
    for row in range (n_test_rows - 10):
        output_pred = predict_next_days (test_copy, plot_column, n_steps, outputs, n_future_iter, file_name, saver, X, y)[-n_future_iter]
        sequence = np.append(sequence, output_pred)
        test_copy.drop(test_copy.index[0], inplace=True)
    return (sequence)
