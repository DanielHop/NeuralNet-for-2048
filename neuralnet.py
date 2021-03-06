import tensorflow as tf

def gen_weights(x, y):
    weights = tf.Variable(tf.truncated_normal([x, y], stddev=1./10))
    return weights

def gen_biases(x):
    bias  = tf.Variable(tf.zeros([x]))
    return bias

def saveTrainNeuralNet(neuralNet, sess, file):
    with open(("models/" + file), 'w') as f:
        f.write('structure: ' + '\n')
        for v in neuralNet.structure:
            f.write(str(v) + " ")

        f.write('\n')
        i = 0
        for weightMatrix in neuralNet.weights:
            f.write('weight matrix ' + str(i) + '\n')
            raw_values = sess.run(weightMatrix)
            for r in raw_values:
                for v in r:
                    f.write(str(v) + " ")
                f.write('\n')
            i += 1


        i = 0
        for biasVector in neuralNet.biases:
            f.write('bias vector ' + str(i) + '\n')
            raw_values = sess.run(biasVector)
            for v in raw_values:
                f.write(str(v))
                f.write(" ")

            f.write('\n')
            i += 1

def loadNeuralNet(file):

    weights = []
    biases = []
    structure = []
    with open(("models/" + file), 'r') as f:

        #Read the structure
        f.readline()
        structure = [int(v) for v in f.readline().split()]
        print(structure)

        #Read the weights
        for i in range(len(structure) - 1):
            f.readline()
            weight_mat = []
            for _ in range(structure[i]):
                weight_vec = [float(v) for v in f.readline().split()]
                if(len(weight_vec) != structure[i + 1]):
                    print("ERROR CORRUPT FILE")
                    return
                weight_mat.append(weight_vec)
            tensor = tf.Variable(weight_mat)
            weights.append(tensor)

        #Read the biases
        for i in range(1, len(structure)):
            f.readline()
            bias_vec = [float(v) for v in f.readline().split()]
            if(len(bias_vec) != structure[i]):
                print("ERROR CORRUPT FILE")
                return
            tensor = tf.Variable(bias_vec)
            biases.append(tensor)

    #Initialize a neural net and return it
    neural_net = TrainNeuralNet(structure, weights, biases)
    return neural_net

class TrainNeuralNet:
    structure = None
    weights   = None
    biases    = None
    values    = None

    n_input_neurons  = 0
    n_output_neurons = 0
    n_hidden_layers  = 0
    n_total_layers   = 0
    epoch_number     = 0

    input_placeholder    = None
    expected_placeholder = None
    output               = None
    softmax_output       = None
    cost                 = None
    train_step           = None

    def __init__(self, structure, weights=None, biases=None):
        #Set the structure array of the neuralnet
        self.structure = structure

        #Derive the number of input neurons, output neurons and the number of hidden layers
        self.n_input_neurons = structure[0]
        self.n_total_layers = len(self.structure)
        self.n_hidden_layers = self.n_total_layers - 2
        self.n_output_neurons = self.structure[self.n_total_layers - 1]

        #Set the placeholers tensors
        self.input_placeholder = tf.placeholder(tf.float32, [None, self.n_input_neurons], "Input_values")
        self.expected_placeholder = tf.placeholder(tf.float32, [None, self.n_output_neurons], "Expected_values")

        #Setup the arrays
        self.weights = []
        self.biases = []
        self.values = []

        #Generate the weights and biases tensors
        if(weights == None or biases==None):
            for i in range(self.n_total_layers - 1):
                self.weights.append(gen_weights(self.structure[i], self.structure[i + 1]))
                self.biases.append(gen_biases(self.structure[i + 1]))
        else:
            if(len(weights)!= self.n_total_layers - 1 or len(biases) != self.n_total_layers - 1):
                print("ERROR WEIGHTS AND OR BIASES DONT MATHC OP WITH THE STRUCTURE")
                return
            self.weights = weights
            self.biases = biases

        #Generate the value tensors

        #The first value is the input, so the input_placeholder
        self.values.append(self.input_placeholder)

        #Generate hidden layer value with prev values in the values list
        for i in range(0, self.n_total_layers - 1):
            prev_layer = self.values[i]

            if(i > 0):
                prev_layer = tf.nn.relu(prev_layer)

            layer_value = tf.matmul(prev_layer, self.weights[i]) + self.biases[i]

            self.values.append(layer_value)

        #Generate the output(softmax of the output layer) and multiply by 100 to get percentages
        self.output = self.values[self.n_total_layers - 1]
        self.softmax_output = tf.nn.softmax(self.output) * 100

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.expected_placeholder))
        self.train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(self.cost)
