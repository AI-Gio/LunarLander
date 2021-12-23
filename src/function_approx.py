import tensorflow.compat.v1 as tf
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import warnings
tf.disable_v2_behavior()
warnings.filterwarnings("ignore", category=UserWarning)


class FunctionApprox:
    """
    The network class of the simulation. Used as to create a policy network instance and a
    target network instance. Contains an artificial neural network from the keras library.
    The output of the neural network is a list of q-values representative of each possible action.
    """
    def __init__(self):

        # Create model
        self.model = Sequential()

        # Add layers
        self.model.add(Dense(input_dim=8, units=1))
        self.model.add(Dense(32, name="1"))
        self.model.add(Dense(64, name="2"))
        self.model.add(Dense(4, name="Output"))

        # Make Adam Optimizer
        adam = tf.keras.optimizers.Adam(learning_rate=0.001, name="Adam")
        loss = tf.keras.losses.MeanSquaredError(name="mean_squared_error")

        # Compile model
        self.model.compile(optimizer=adam, loss=loss)

    def q_values(self, states: list) -> np.array:
        """
        Feeds list of states in model to predict and gives a list of
        Q-values, one for each action, in return.
        :param states:
        :return: np.array() of predictions
        """
        predictions = self.model.predict(np.array(states))
        return predictions

    def save_network(self, filename):
        """
        Save the model at [filename]
        """
        self.model.save(f"{filename}.h5")

    def load_network(self, filename):
        """
        Load model by overwriting current model with keras.model object
        """
        self.model = tf.keras.models.load_model(f"{filename}.h5")

    def train(self, x, y, batch_size, epochs: int, verbose: bool):
        """
        Train the model with the given params
        :param x: set with train data
        :param y: set with labels
        :param batch_size: how much of the data should be used to train on
        :param epochs: how many epochs the model runs
        :param verbose: show training progress bar or not
        """
        self.model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs, verbose=verbose)

    def set_weights(self, weights: np.array):
        """
        Set the weights of the model layer by layer
        :param weights: all weights of model
        """
        layers = self.model.layers
        for i, lw in enumerate(layers):
            lw.set_weights(weights[i])

    def get_weights(self):
        """
        Get all weights from each layers in model
        :return: 2d numpy matrix containing all the weights
        """
        layers = []
        for layer in self.model.layers:
            layers.append(layer.get_weights())
        return np.array(layers, dtype=object)
