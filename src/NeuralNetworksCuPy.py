"""
Completed ArtificialNeuralNetworks class exploiting CuPy's GPU-accelerated
vectorization. This serves as a drop-in, highly parallelized alternative to the NumPy version.
"""
import cupy as cp


class ArtificialNeuralNetworks:
    def __init__(self, layer_shapes, activation_function='relu', final_layer_activation='sigmoid'):
        """
        weights[0] => neurons between layers 0 and 1. rows: right layer neurons,
        columns: left layer. weights[0][2][5] => layer 0's 5'th neuron with layer 1's 2nd neuron
        """
        self.zs = None
        rng = cp.random.default_rng()

        self.layers = list()  # will hold neurons activations e.g. layers[0] => neurons in first layer
        for shape in layer_shapes:
            self.layers.append(None)

        self.biases = list()
        for i in range(1, len(layer_shapes)):   # layer 0 doesn't have bias
            # update: changed bias to a 2D column vector to exploit broadcasting
            self.biases.append(rng.standard_normal((layer_shapes[i],1), dtype=cp.float32))

        if activation_function=='relu':
            self.activation_f = self._relu
            self.activation_f_derivative = self._relu_derivative
        elif activation_function == 'sigmoid':
            self.activation_f = self._sigmoid
            self.activation_f_derivative = self._sigmoid_derivative
        else:
            raise ValueError(f"Invalid func: '{activation_function}'. ""Expected 'relu' or 'sigmoid'.")

        if final_layer_activation=='relu':
            self.f_activation_f = self._relu
            self.f_activation_f_derivative = self._relu_derivative
        elif final_layer_activation == 'sigmoid':
            self.f_activation_f = self._sigmoid
            self.f_activation_f_derivative = self._sigmoid_derivative
        else:
            raise ValueError(f"Invalid func: '{final_layer_activation}'. ""Expected 'relu' or 'sigmoid'.")

        self.weights = list()
        for i in range(len(self.layers)-1):
            shape = (layer_shapes[i+1], layer_shapes[i])
            # Standard practice: scale weights down so gradients don't explode
            input_size = layer_shapes[i]
            he_scale = cp.sqrt(2.0 / input_size)
            xavier_scale = cp.sqrt(1.0 / input_size)
            scale = he_scale if activation_function=='relu' else xavier_scale
            scale = scale if i!=len(self.layers)-2 else he_scale if final_layer_activation=='relu' else xavier_scale
            self.weights.append(rng.standard_normal(shape, dtype=cp.float32) * scale)


    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + cp.exp(-cp.clip(z, -500, 500)))

    @staticmethod
    def _sigmoid_derivative(z):
        s = 1 / (1 + cp.exp(-cp.clip(z, -500, 500)))
        return s * (1 - s)

    @staticmethod
    def _relu(z):
        return cp.maximum(0, z)

    @staticmethod
    def _relu_derivative(z):
        return cp.where(z > 0, 1.0, 0.0)


    def _predict_probabilities(self, x):
        # update: convert CPU/NumPy array to GPU/CuPy array seamlessly
        x = cp.asarray(x)
        self.layers[0] = x.T
        self.zs = list()
        for i in range(0, len(self.layers)-2):
            z = self.weights[i] @ self.layers[i]+self.biases[i] # shape => (n(l), m). bias is broadcasted automatically
            self.zs.append(z)
            self.layers[i+1] = self.activation_f(z)
        i = len(self.layers)-2
        z = self.weights[i] @ self.layers[i] + self.biases[i]
        self.zs.append(z)
        self.layers[i + 1] = self.f_activation_f(z)
        results = self.layers[-1].T     # redo the transpose
        return results

    def predict_probabilities(self, x):
        return self._predict_probabilities(x).get()     # returns cpu object. couldn't incorporate this into the main predict because it's used in gradient step

    def fit(self, x, y, learning_rate=0.01, epochs=100, batch_size=None):
        """
        :param x: x train
        :param y: y train
        :param learning_rate:
        :param epochs: n-gradient descent repetitions
        :param batch_size: batch size for stochastic gd
        :return: none
        """
        # Safely convert incoming training data to GPU arrays
        x_train = cp.asarray(x)
        y_train = cp.asarray(y)
        self.__batched_stochastic_gradient_descent(x_train, y_train, learning_rate, epochs, batch_size)


    def __batched_stochastic_gradient_descent(self, x_train, y_train, learning_rate, epochs, batch_size):
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        m = x_train.shape[0]
        if batch_size is None or batch_size > m:
            batch_size = m


        for epoch in range(epochs):
            epoch_loss = 0.0

            # 1. Shuffle the dataset at the start of every epoch to maintain stochastic randomness
            indices = cp.arange(m)
            cp.random.shuffle(indices)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]

            for start_idx in range(0, m, batch_size):   # Iterate through the dataset in mini-batch chunks
                end_idx = min(start_idx + batch_size, m)
                x = x_shuffled[start_idx:end_idx]
                y = y_shuffled[start_idx:end_idx]
                epoch_loss += self.__gradient_descent_step(x, y, learning_rate)

            if epoch % 10 == 0:
                # Update: Wrap cp.sum in float()
                print(f'{epoch}\'th repetition. Mean Squared Error: {float(cp.sum(epoch_loss)) / m:.6f}')


    def __gradient_descent_step(self, x_train, y_train, learning_rate):
        prediction = self._predict_probabilities(x_train)    # shape(y, m)
        batch_loss = cp.sum(cp.power(prediction - y_train, 2), axis=0)   # shape => (1, m)

        weights_gradient = list()  # list of changes for each layer's weights
        bias_gradient = list()

        # find gradient for the final layer first, then iterate to the first layer(final layer's bias and weights entering final layer)
        # for the last layer: del cost/a(l) => (2*(prediction-data)) | del(a)/z = sigmoid'(z) | del zj(l)/wjk => ak(l-1)
        delta = (2*(prediction-y_train).T) * (self.f_activation_f_derivative(self.zs[-1]))

        # Update: The total bias gradient is the sum of the errors across all samples in the batch.
        # We need to collapse the sample columns by summing across axis=1, while keeping the dimensions 2D so it stays a column vector:
        bias_gradient.insert(0, cp.sum(delta, axis=1, keepdims=True))
        weights_gradient.insert(0, delta @ self.layers[-2].T)       # update

        # loop back to find other layers tuning vals

        for i in range(len(self.layers) - 2, 0, -1):    # find weights between layers i-1, i gradients.
            # Passing delta backwards
            delta = (self.weights[i].T @ delta) * self.activation_f_derivative(self.zs[i - 1])
            bias_gradient.insert(0, cp.sum(delta, axis=1, keepdims=True))

            # update: delta matrix has the shape (neurons_out, samples), previous layer activations matrix has the shape (neurons_in, samples).
            # by doing delta @ a(l-1)T we sum up the gradients across all samples and fill weights_gradient[j][k]
            weights_gradient.insert(0, delta @ self.layers[i - 1].T)


        # apply gradient values to weights and biases. update: average over all samples
        m = x_train.shape[0]
        for i in range(len(self.weights)):
            self.weights[i] -= (weights_gradient[i] / m) * learning_rate
        for i in range(len(self.biases)):
            self.biases[i] -= (bias_gradient[i] / m) * learning_rate

        return batch_loss