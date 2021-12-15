from tensorflow import keras


class FunctionApprox:
    def __init__(self):
        # Create model
        self.model = keras.Sequential()

        # Input layer
        self.model.add(keras.Input(8,))

        # Add layers
        self.model.add(keras.layer.Dense(32, name="1"))
        self.model.add(keras.layer.Dense(32, name="2"))

        # Make Adam Optimizer
        adam = keras.optimizers.Adam(learning_rate=0.001, name="Adam")
        loss = keras.optimizers.RMSprop(learning_rate=0.001, name="RMSprop")

        # Compile model
        self.model.compile(optimizer=adam, loss=loss)

    def q_values(self, states: list):
        """
        Feeds list of states in model to predict
        :param states:
        :return: np.array() of predictions
        """
        preds = []
        for state in states:
            preds.append(self.model.predict(state))
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
        keras.load_model("CurrentModel.h5")

    def train(self, x, y, batch_size, epochs: int, verbose: bool, validation_split: float, shuffle: bool):
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
        self.model.fit(x=x,y=y, batch_size=batch_size, epochs=epochs, verbose=verbose,
                       validation_split=validation_split, shuffle=shuffle)

    def set_weights(self, l_w: list, layer: int):
        """
        Set the weights of the model
        :param l_w: list with weights (length must correspond to same amount of neurons in the layer)
        :param layer: integer that corresponds to what layer in the model the weights should be changed
        """
        layers = self.model.layer
        for l in layers:
            if int(l.name) == layer:
                self.model.set_weights(l_w)

    def get_weights(self):
        """
        Get weigths from layers in model
        :return: 2d list with np.arrays
        """
        layers = []
        for layer in self.model.layers:
            layers.append(layer.get_weights())
        return layers



