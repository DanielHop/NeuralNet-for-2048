#from __future__ import print_function
import tensorflow as tf
import numpy as np

import ctypes
import time
import os
import gamemanager as manager
import neuralnet

# load the algorithm for getting the best move
for suffix in ['so', 'dll', 'dylib']:
    dllfn = 'bin/2048.' + suffix
    if not os.path.isfile(dllfn):
        continue
    ailib = ctypes.CDLL(dllfn)
    break
else:
    print("Couldn't find 2048 library bin/2048.{so,dll,dylib}! Make sure to build it first.")
    exit()

# Define the find_best_move() function
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

def find_heighest(board):
    heighest = 0
    for x in board:
        for y in x:
            if(y > heighest):
                heighest = y
    return heighest

def transpose_board(board, heighest):
    b2 = []

    for x in range(len(board)):
        for y in range(len(board[x])):
            b2.append(float(board[x][y] / heighest))

    return [b2]

def one_hot(move):
    # up - down - left - right
    return [[[1, 0, 0, 0]],[[0, 1, 0, 0]],[[0, 0, 1, 0]],[[0, 0, 0, 1]]][move]

def log2(x):
    if(x==0):
        return 0
    elif(x == 2):
        return 1
    elif(x==4):
        return 2
    elif(x==8):
        return 3
    elif(x==16):
        return 4
    elif(x==32):
        return 5
    elif(x==64):
        return 6
    elif(x==128):
        return 7
    elif(x==256):
        return 8
    elif(x==512):
        return 9
    elif(x==1024):
        return 10
    elif(x==2048):
        return 11
    elif(x==4096):
        return 12
    elif(x==8192):
        return 13
    else:
        return 14

def log2board(board):
    returnboard = []
    for i in board:
        g = [log2(i[j]) for j in range(4)]
        returnboard.append(g)
    return returnboard

def splitboard(board):
    return board.tolist()


if __name__ == "__main__":

    ailib.init_tables()

    ailib.find_best_move.argtypes = [ctypes.c_uint64]
    ailib.score_toplevel_move.argtypes = [ctypes.c_uint64, ctypes.c_int]
    ailib.score_toplevel_move.restype = ctypes.c_float

    it = 18000

    if it == 0:
        net_8 =  neuralnet.TrainNeuralNet([16, 256, 2400, 1600, 800, 400, 200, 200, 100, 4])
        net_9 =  neuralnet.TrainNeuralNet([16, 256, 2400, 1600, 800, 400, 400, 200, 200, 100, 4])
        net_10 = neuralnet.TrainNeuralNet([16, 256, 2400, 1600, 800, 400, 400, 400, 200, 200, 100, 4])
    else:
        net_8 = neuralnet.loadNeuralNet('net_8_' + str(it))
        net_9 = neuralnet.loadNeuralNet('net_9_' + str(it))
        net_10 = neuralnet.loadNeuralNet('net_10_' + str(it))

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        currGame = manager.Game2048(4)
        reset = manager.Game2048(4)
        r = 0
        j = 1
        done = False
        for i in range(it, 300000):
            splitgrid = splitboard(currGame.grid_)

            heighest = find_heighest(splitgrid)
            if heighest == 4096 or done:
                currGame = reset.clone()
                heighest = 1
                r += 1
                done = False

            #Get the best move
            m_number = find_best_move(log2board(splitgrid))


            #Transpose board
            transposed_board = transpose_board(splitgrid, heighest)

            #get the one hot
            expected_values = one_hot(m_number)

            #Train data on move
            #i_v = m.data & e_v = best_move
            sess.run(net_8.train_step, feed_dict={net_8.input_placeholder:  transposed_board, net_8.expected_placeholder: expected_values})
            sess.run(net_9.train_step, feed_dict={net_9.input_placeholder:  transposed_board, net_9.expected_placeholder: expected_values})
            sess.run(net_10.train_step, feed_dict={net_10.input_placeholder: transposed_board, net_10.expected_placeholder: expected_values})

            old_game = currGame.clone()
            new_game = currGame.clone()

            new_game.move(manager.movename(m_number))

            currGame.grid_ = new_game.grid_
            if(currGame != old_game):
                currGame.evolve()
            else:
                done = True

            print(i, r)
            if (i + 1) % 1000 == 0:
                print(sess.run(net_8.cost, feed_dict={net_8.input_placeholder:  transposed_board, net_8.expected_placeholder: expected_values}))
                print(sess.run(net_9.cost, feed_dict={net_9.input_placeholder:  transposed_board, net_9.expected_placeholder: expected_values}))
                print(sess.run(net_10.cost, feed_dict={net_10.input_placeholder:  transposed_board, net_10.expected_placeholder: expected_values}))

                print(heighest)

                print("saving")
                neuralnet.saveTrainNeuralNet(net_8, sess, ('net_8_' + str(i + 1)))
                neuralnet.saveTrainNeuralNet(net_9, sess, ('net_9_' + str(i + 1)))
                neuralnet.saveTrainNeuralNet(net_10, sess, ('net_10_' + str(i + 1)))
