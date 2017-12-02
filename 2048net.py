#from __future__ import print_function
import tensorflow as tf
import numpy as np

import ctypes
import time
import os

import gamemanager as manager
import neuralnet

# Enable multithreading?

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

MULTITHREAD = True

if MULTITHREAD:
    from multiprocessing.pool import ThreadPool
    pool = ThreadPool(4)
    def score_toplevel_move(args):
        return ailib.score_toplevel_move(*args)

    def find_best_move(m):
        board = manager.to_c_board(m)

        #print_board(to_val(m))

        scores = pool.map(score_toplevel_move, [(board, move) for move in range(4)])
        bestmove, bestscore = max(enumerate(scores), key=lambda x:x[1])
        if bestscore == 0:
            return -1
        return bestmove
else:
    def find_best_move(m):
        board = manager.to_c_board(m)
        return ailib.find_best_move(board)


net = neuralnet.NeuralNet([2, 10, 10, 10, 2])
#Training data
XOR_X = [[0,0],[0,1],[1,0],[1,1]]
XOR_Y = [[1,0],[0,1],[0,1],[1,0]]


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	currGame = manager.Map(4, 4, 2)
	reset = manager.Map(4, 4, 2)
	r = 0
	for i in range(3000):
		p = currGame.get_random_empty()
		if p == 0:
			currGame = reset
			r += 1

		#Get the best move
		m = manager.movename(find_best_move(currGame.data))
		print("BEST MOVE: ", m)

		#Train data on move
		#i_v = m.data & e_v = best_move
		sess.run(net.train_step, feed_dict={net.input_placeholder: XOR_X, net.expected_placeholder: XOR_Y})

		if(currGame.move(m) == 1):
			currGame.set_cell(p[0], p[1], manager.new_cell_value())

		currGame.print_map()


		if (i + 1) % 1000 == 0:
			for x_input in [[0, 0], [0, 1], [1, 0], [1, 1]]:
				print (x_input, sess.run(net.softmax_output, feed_dict={net.input_placeholder: XOR_X, net.expected_placeholder: XOR_Y}))
			print('Epoch ', i + 1)
			#print('Hypothesis ', sess.run(output, feed_dict={i_n: XOR_X, o_n: XOR_Y}))
			#print('Theta1 ', sess.run(weight1))
			#print('Theta2 ', sess.run(weight2))
			print('cost ', sess.run(net.cost, feed_dict={net.input_placeholder: XOR_X, net.expected_placeholder: XOR_Y}))
	print(r)
