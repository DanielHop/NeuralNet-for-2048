#from __future__ import print_function
import tensorflow as tf
import numpy as np

import ctypes
import time
import os

import game
import neuralnet

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

net = neuralnet.NeuralNet([2, 10, 10, 10, 2])
#Training data
XOR_X = [[0,0],[0,1],[1,0],[1,1]]
XOR_Y = [[1,0],[0,1],[0,1],[1,0]]


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	currGame = game.Map(4, 4, 2)
	reset = game.Map(4, 4, 2)
	r = 0
	for i in range(3000):
		#p = currGame.get_random_empty()
		#if p == 0:
			#currGame = reset
			#r += 1

		#Get the best move
		#m = movename(get_best_move(currGame.data))
		#print("BEST MOVE: ", m)

		#Train data on move
		#i_v = m.data & e_v = best_move
		sess.run(net.train_step, feed_dict={net.input_placeholder: XOR_X, net.expected_placeholder: XOR_Y})

		#if(currGame.move(m) == 1):
			#currGame.set_cell(p[0], p[1], game.new_cell_value())

		#currGame.print_map()


		if (i + 1) % 1000 == 0:
			for x_input in [[0, 0], [0, 1], [1, 0], [1, 1]]:
				print (x_input, sess.run(net.softmax_output, feed_dict={net.input_placeholder: XOR_X, net.expected_placeholder: XOR_Y}))
			print('Epoch ', i + 1)
			#print('Hypothesis ', sess.run(output, feed_dict={i_n: XOR_X, o_n: XOR_Y}))
			#print('Theta1 ', sess.run(weight1))
			#print('Theta2 ', sess.run(weight2))
			print('cost ', sess.run(net.cost, feed_dict={net.input_placeholder: XOR_X, net.expected_placeholder: XOR_Y}))
	print(r)
