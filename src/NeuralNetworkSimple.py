import numpy as np


class ArtificialNeuralNetworks:
    def __init__(self, layer_shapes, activation_function='relu', final_layer_activation='sigmoid'):
        """
        weights[0] => neurons between layers 0 and 1. rows: right layer neurons,
        columns: left layer. weights[0][2][5] => layer 0's 5'th neuron with layer 1's 2nd neuron
        """
        # random generator
        self.zs = None
        rng = np.random.default_rng()

        self.layers = list()  # will hold neurons activations e.g. layers[0] => neurons in first layer
        for shape in layer_shapes:
            self.layers.append(np.zeros(shape, dtype=np.float32))   # todo: 16 bit quantization?

        self.biases = list()
        for i in range(1, len(layer_shapes)):   # layer 0 doesn't have bias
            self.biases.append(rng.standard_normal(layer_shapes[i], dtype=np.float32))  # assign a bias to every neuron

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

        self.weights = list()  # 0: layer 1-2 | 1: layer 2-3 ...
        for i in range(len(self.layers)-1):
            shape = (self.layers[i+1].size, self.layers[i].size)

            # Standard practice: scale weights down so gradients don't explode
            # for ReLU: i use He initialization. Scale by radical(2/n(in))
            # for sigmoid: Xavier initialization. Scale by radical(1/n(in)) where n(in) is the size of incoming layer
            input_size = self.layers[i].size
            he_scale = np.sqrt(2.0 / input_size)
            xavier_scale = np.sqrt(1.0 / input_size)
            scale = he_scale if activation_function=='relu' else xavier_scale
            scale = scale if i==len(self.layers)-2 else he_scale if final_layer_activation=='relu' else xavier_scale    # not ideal to have many branches in a loop,
            # however we're looping over depth of the network which is going to be a small constant so it's fine.
            self.weights.append(rng.standard_normal(shape, dtype=np.float32)*scale)


    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    @staticmethod
    def _sigmoid_derivative(z):
        s = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        return s * (1 - s)

    @staticmethod
    def _relu(z):
        return np.maximum(0, z)

    @staticmethod
    def _relu_derivative(z):
        return np.where(z > 0, 1, 0)


    def predict_probabilities(self, x):
        if x.ndim == 1:     # vector(used in fit)
            self.layers[0] = x
            self.zs = list()
            for i in range(0, len(self.layers)-2):  # except for final layer with different activation
                z = self.weights[i] @ self.layers[i]+self.biases[i]
                self.zs.append(z)
                self.layers[i+1] = self.activation_f(z)
            # for final layer with different activation
            i = len(self.layers)-2
            z = self.weights[i] @ self.layers[i] + self.biases[i]
            self.zs.append(z)
            self.layers[i + 1] = self.f_activation_f(z)
            return self.layers[-1].copy()


        else:       # dataset(usual stuff)
            num_samples = x.shape[0]
            results = np.zeros((num_samples, self.layers[-1].size), dtype=np.float32)
            for j in range(x.shape[0]):         # row samples
                # provided a value for each neuron in the first layer, we have to calculate final layer's neurons values
                # e.g. 1'st neuron on layer2 (a0(1)) => (w0,0,0*a0(0) + w0,0,1*a1(0) etc.)
                self.layers[0] = x[j]
                self.zs = list()
                for i in range(0, len(self.layers)-2):
                    z = self.weights[i] @ self.layers[i]+self.biases[i]
                    self.zs.append(z)
                    self.layers[i+1] = self.activation_f(z)
                i = len(self.layers)-2
                z = self.weights[i] @ self.layers[i] + self.biases[i]
                self.zs.append(z)
                self.layers[i + 1] = self.activation_f(z)
                results[j] = self.layers[-1].copy()

        return results


    def fit(self, x, y, method='standard', activation_function='relu', learning_rate=0.01, epochs=100, batch_size=None):
        """
        :param x: x train
        :param y: y train
        :param method: 'standard' or 'stochastic'
        :param activation_function: 'relu' or 'sigmoid'
        :param learning_rate:
        :param epochs: n-gradient descent repetitions
        :param batch_size: batch size for stochastic gd
        :return: none
        """
        if method == 'standard':
            self.__online_stochastic_gradient_descent(x, y, learning_rate, epochs)
        elif method == 'stochastic':
            self.__batched_stochastic_gradient_descent(x, y, learning_rate, epochs, batch_size)


    def __batched_stochastic_gradient_descent(self, x_train, y_train, learning_rate, epochs, batch_size):
        # todo
        pass


    def __online_stochastic_gradient_descent(self, x_train, y_train, learning_rate, epochs):
        for epoch in range(epochs):
            epoch_loss = 0.0

            for x, y in zip(x_train, y_train):
                prediction = self.predict_probabilities(x)   # z
                epoch_loss += np.sum(np.power(prediction - y, 2))

                weights_gradient = list()  # list
                bias_gradient = list()

                # find gradient for the final layer first, then iterate to the first layer(final layer's bias and weights entering final layer)
                # for the last layer: del cost/a(l) => (2*(prediction-data)) | del(a)/z = sigmoid'(z) | del zj(l)/wjk => ak(l-1)
                delta = (2*(prediction-y)) * (self.f_activation_f_derivative(self.zs[-1]))
                bias_gradient.insert(0, delta)
                # del(Zj)/Wjk for any j,k => a(l-1)k => (j,k) matrix
                weights_gradient.insert(0,np.outer(delta, self.layers[-2]))     # essentially, the same as doing delta@temp where temp is j*k matrix where temp[j][k] is a(l-1)k



                # loop back to find other layers tuning vals

                # delta always holds del(cost)/del(a(l+1)) * del(a(l+1))/del(z(l+1)) which is del(cost)/del(z+1)
                # so to calculate del(cost)/del(z(l)) we just need delta * del(z+1)/del(z(l)) which is w(l+1) * sigmoid derivative?
                # after we have del(cost)/del(z(l)) we store it in delta, then del(cost)/del(w(l-1)) = delta * a(l-2) and bias is juts delta
                for i in range(len(self.layers) - 2, 0, -1):    # find weights between layers i-1, i gradients.
                    # Passing delta backwards
                    delta = (self.weights[i].T @ delta) * self.activation_f_derivative(self.zs[i - 1])
                    bias_gradient.insert(0, delta)
                    weights_gradient.insert(0, np.outer(delta, self.layers[i-1]))


                # apply gradient values to weights and biases
                for i in range(len(self.weights)):
                    self.weights[i] = self.weights[i]-weights_gradient[i]*learning_rate
                for i in range(len(self.biases)):
                    self.biases[i] = self.biases[i] - bias_gradient[i]*learning_rate

            if epoch%10==0:
                print(f'{epoch}\'th repetition. Total Squared Error: {epoch_loss:.4f}')



# todo: still some nuances left