#from __future__ import print_function
import tensorflow as tf
import numpy as np
import game
import ctypes
import time
import os

#For testing purposes
n_input_neurons = 2
n_output_neurons = 2

n_neurons = np.array([n_input_neurons, 10, n_output_neurons])

i_v = tf.placeholder(tf.float32, [None, n_input_neurons], "Input_values")
e_v = tf.placeholder(tf.float32, [None, n_output_neurons], "Expected_values")

#define helper functions for generating weights and biases
def gen_weights(x, y):
	weights = tf.Variable(tf.truncated_normal([x, y], stddev=1./tf.sqrt(2.)))
	return weights

def gen_biases(x):
	bias  = tf.Variable(tf.zeros([x]))
	return bias

#initialize weights and biases in arrays
weights = []
biases = []

for i in range(n_neurons.size - 1):
	weights.append(gen_weights(n_neurons[i], n_neurons[i + 1]))
	biases.append(gen_biases(n_neurons[i + 1]))

values = []

#Generate first layer value with the i_v
values.append(i_v)
#Generate hidden layer value with prev values in the values list
for i in range(0, n_neurons.size - 1):
	prev_layer = values[i]
	
	if(i > 0):
		prev_layer = tf.nn.relu(values[i])
	
	layer_value = tf.matmul(prev_layer, weights[i]) + biases[i]
	
	values.append(layer_value)


#Generate last layer value without the ReLU
output = values[n_neurons.size - 1]
prop = tf.nn.softmax(output) * 100

#Set the cost and trainign step
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=e_v))
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cost)

#Training data
XOR_X = [[0,0],[0,1],[1,0],[1,1]]
XOR_Y = [[1,0],[0,1],[0,1],[1,0]]

# Enable multithreading?
MULTITHREAD = True

for suffix in ['so', 'dll', 'dylib']:
    dllfn = 'bin/2048.' + suffix
    if not os.path.isfile(dllfn):
        continue
    ailib = ctypes.CDLL(dllfn)
    break
else:
    print("Couldn't find 2048 library bin/2048.{so,dll,dylib}! Make sure to build it first.")
    exit()

ailib.init_tables()

ailib.find_best_move.argtypes = [ctypes.c_uint64]
ailib.score_toplevel_move.argtypes = [ctypes.c_uint64, ctypes.c_int]
ailib.score_toplevel_move.restype = ctypes.c_float

def to_c_board(m):
    board = 0
    i = 0
    for row in m:
        for c in row:            
            board |= c << (4*i)
            i += 1
    return board

def print_board(m):
    for row in m:
        for c in row:
            print('%8d' % c, end=' ')
        print()

def _to_val(c):
    if c == 0: return 0
    return 2**c

def to_val(m):
    return [[_to_val(c) for c in row] for row in m]

def _to_score(c):
    if c <= 1:
        return 0
    return (c-1) * (2**c)

def to_score(m):
    return [[_to_score(c) for c in row] for row in m]

if MULTITHREAD:
    from multiprocessing.pool import ThreadPool
    pool = ThreadPool(4)
    def score_toplevel_move(args):
        return ailib.score_toplevel_move(*args)

    def find_best_move(m):
        board = to_c_board(m)

        #print_board(to_val(m))

        scores = pool.map(score_toplevel_move, [(board, move) for move in range(4)])
        bestmove, bestscore = max(enumerate(scores), key=lambda x:x[1])
        if bestscore == 0:
            return -1
        return bestmove
else:
    def find_best_move(m):
        board = to_c_board(m)
        return ailib.find_best_move(board)

def get_best_move(m):
	return find_best_move(m)

def movename(move):
    return ['u', 'd', 'l', 'r'][move]

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	
	currGame = game.Map(4, 4, 2)
	reset = game.Map(4, 4, 2)
	r = 0
	for i in range(1000):
		p = currGame.get_random_empty()
		if p == 0:
			currGame = reset
			r += 1

		#Get the best move
		m = movename(get_best_move(currGame.data))
		print("BEST MOVE: ", m)

		#Train data on move
		#i_v = m.data & e_v = best_move
		#sess.run(train_step, feed_dict={i_v: XOR_X, e_v: XOR_Y})
		
		if(currGame.move(m) == 1):
			currGame.set_cell(p[0], p[1], game.new_cell_value())

		currGame.print_map()


		if i % 1000 == 0:
			for x_input in [[0, 0], [0, 1], [1, 0], [1, 1]]:
				print (x_input, sess.run(prop, feed_dict={i_v: [x_input]}))
			print('Epoch ', i)
			#print('Hypothesis ', sess.run(output, feed_dict={i_n: XOR_X, o_n: XOR_Y}))
			#print('Theta1 ', sess.run(weight1))
			#print('Theta2 ', sess.run(weight2))
			print('cost ', sess.run(cost, feed_dict={i_v: XOR_X, e_v: XOR_Y}))
	print(r)