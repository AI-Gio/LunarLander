import tensorflow as tf
import numpy as np

class FunctionApprox:
    def __init__(self):
        # Create model
        self.model = tf.keras.Sequential()


        # Add layers
        self.model.add(tf.keras.layers.Dense(8,))
        self.model.add(tf.keras.layers.Dense(32, name="1"))
        self.model.add(tf.keras.layers.Dense(32, name="2"))
        self.model.add(tf.keras.layers.Dense(4, name="Output"))

        # Make Adam Optimizer
        adam = tf.keras.optimizers.Adam(learning_rate=0.001, name="Adam")
        loss = tf.keras.optimizers.RMSprop(learning_rate=0.001, name="mean_squared_error")

        # Compile model
        self.model.compile(optimizer=adam, loss=loss)

    def q_values(self, states: list):
        """
        Feeds list of states in model to predict
        :param states:
        :return: np.array() of predictions
        """
        # print(states)
        # print(np.array([np.array(states)]))
        preds = self.model.predict(np.array([np.array(states)]))[0]

        # for state in states:
        #     print(state)
        #     preds.append(self.model.predict(state))
        # if len(preds) < 2:
        #     return preds[0]
        return preds

    def save_network(self):
        """
        Save model
        """
        self.model.save("CurrentModel.h5")

    def load_network(self):
        """
        Load model
        """
        tf.keras.load_model("CurrentModel.h5")

    def train(self, x, y, batch_size, epochs: int, verbose: bool):
        """
        Train the model with the given params
        :param x: set with train data
        :param y: set with labels
        :param batch_size: how much of the data should be used to train on
        :param epochs: how many epochs the model runs
        :param verbose: show training progress or not
        :param validation_split: how much of the data will be kept on the side to test trained model later on
        :param shuffle: shuffle data
        """
        self.model.fit(x=x,y=y, batch_size=batch_size, epochs=epochs, verbose=verbose)

    def set_weights(self, weights: np.array):
        """
        Set the weights of the model
        :param weights: all weights of model
        :param layer: integer that corresponds to what layer in the model the weights should be changed
        """
        layers = self.model.layers
        for i, lw in enumerate(layers):
            self.model.lw.set_weights(weights[i])

    def get_weights(self):
        """
        Get weigths from layers in model
        :return: 2d np.array with np.arrays
        """
        layers = np.array([])
        for layer in self.model.layers:
            np.append(layers, layer.get_weights())
        return layers



