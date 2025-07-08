
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as kb

def learning_rate_step_decay(epoch, lr, step=50, initial_power=-4): 
    """
    The learning rate begins at 10^initial_power,
    and decreases by a factor of 10 every `step` epochs.
    """
    ##
    num = epoch // step
    lrate = 10 ** (initial_power - num)
    print("Learning rate plan for epoch {} is {}.".format(epoch + 1, 1.0 * lrate))
    return np.float(lrate)

def learning_rate_step_decay2(epoch, lr, step=50, initial_power=-4):
    """
    The learning rate begins at 10^initial_power,
    and decreases by a factor of 10 every `step` epochs.
    """
    ##
    num = epoch // step
    lrate = 10 ** (initial_power - num / 2)
    print("Learning rate plan for epoch {} is {}.".format(epoch + 1, 1.0 * lrate))
    return np.float(lrate)


def learning_rate_step_decay_classic(epoch, lr, decay = 0.01, initial_power=-4, start_epoch = 0):
    """
    The learning rate begins at 10^initial_power,
    and decreases by a factor of 10 every `step` epochs.
    """
    ##

    lrate = (1/ (1 + decay * (epoch + start_epoch))) * (10 ** initial_power)

    print("Learning rate plan for epoch {} is {}.".format(epoch + 1, 1.0 * lrate))
    return np.float(lrate)


def weighted_loss(y_true, y_pred, weights = [1,1,1]):

    # Calculate the MSE for each output
    loss_x1 = tf.keras.losses.mean_squared_error(y_true[0], y_pred[0])
    loss_x2 = tf.keras.losses.mean_squared_error(y_true[1], y_pred[1])
    loss_x3 = tf.keras.losses.mean_squared_error(y_true[2], y_pred[2])

    # Calculate the weighted sum of the MSE
    final_loss = tf.reduce_sum([weights[i] * [loss_x1, loss_x2, loss_x3][i] for i in range(3)])

    return final_loss