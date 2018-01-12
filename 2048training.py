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
            if(heighest == 0):
                b2.append(0)
            else:
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

def average(numbers):
    total = 0
    for i in numbers:
        total += i
    total = total/len(numbers)
    return total

if __name__ == "__main__":

    ailib.init_tables()

    ailib.find_best_move.argtypes = [ctypes.c_uint64]
    ailib.score_toplevel_move.argtypes = [ctypes.c_uint64, ctypes.c_int]
    ailib.score_toplevel_move.restype = ctypes.c_float

    it = 305000

    if it == 0:
        net_8_1 =  neuralnet.TrainNeuralNet([16, 256, 2400, 1600, 800, 400, 200, 200, 100, 4])
        net_8_2 =  neuralnet.TrainNeuralNet([16, 256, 2400, 1600, 800, 400, 200, 200, 100, 4])
        net_9_1 =  neuralnet.TrainNeuralNet([16, 256, 2400, 1600, 800, 400, 400, 200, 200, 100, 4])
        net_9_2 =  neuralnet.TrainNeuralNet([16, 256, 2400, 1600, 800, 400, 400, 200, 200, 100, 4])
        net_10_1 = neuralnet.TrainNeuralNet([16, 256, 2400, 1600, 800, 400, 400, 400, 200, 200, 100, 4])
        net_10_2 = neuralnet.TrainNeuralNet([16, 256, 2400, 1600, 800, 400, 400, 400, 200, 200, 100, 4])
    else:
        net_8_1 =  neuralnet.loadNeuralNet('net_8_1_' + str(it))
        net_8_2 =  neuralnet.loadNeuralNet('net_8_2_' + str(it))
        net_9_1 =  neuralnet.loadNeuralNet('net_9_1_' + str(it))
        net_9_2 =  neuralnet.loadNeuralNet('net_9_2_' + str(it))
        net_10_1 = neuralnet.loadNeuralNet('net_10_1_' + str(it))
        net_10_2 = neuralnet.loadNeuralNet('net_10_2_' + str(it))

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        currGame = manager.Game2048(4)
        reset = manager.Game2048(4)
        r = 0
        j = 1
        done = False

        cost_8 = []
        cost_9 = []
        cost_10 = []
        for i in range(it, 600000):
            splitgrid = log2board(splitboard(currGame.grid_))

            heighest = find_heighest(splitgrid)
            if heighest == 13 or done:
                currGame = reset.clone()
                splitgrid = splitboard(currGame.grid_)
                heighest = find_heighest(splitgrid)
                r += 1
                done = False

            #Get the best move
            m_number = find_best_move(splitgrid)


            #Transpose board
            transposed_board = transpose_board(splitgrid, heighest)

            #get the one hot
            expected_values = one_hot(m_number)

            #Train data on move
            if(heighest > 0):
                _, c8_1 = sess.run([net_8_1.train_step, net_8_1.cost], feed_dict={net_8_1.input_placeholder:  transposed_board, net_8_1.expected_placeholder: expected_values})
                _, c8_2 = sess.run([net_8_2.train_step, net_8_2.cost], feed_dict={net_8_2.input_placeholder:  transposed_board, net_8_2.expected_placeholder: expected_values})
                _, c9_1 = sess.run([net_9_1.train_step, net_9_1.cost], feed_dict={net_9_1.input_placeholder:  transposed_board, net_9_1.expected_placeholder: expected_values})
                _, c9_2 = sess.run([net_9_2.train_step, net_9_2.cost], feed_dict={net_9_2.input_placeholder:  transposed_board, net_9_2.expected_placeholder: expected_values})
                _, c10_1 = sess.run([net_10_1.train_step, net_10_1.cost], feed_dict={net_10_1.input_placeholder: transposed_board, net_10_1.expected_placeholder: expected_values})
                _, c10_2 = sess.run([net_10_2.train_step, net_10_2.cost], feed_dict={net_10_2.input_placeholder: transposed_board, net_10_2.expected_placeholder: expected_values})


            cost_8.append(c8_1)
            cost_8.append(c8_2)
            cost_9.append(c9_1)
            cost_9.append(c9_2)
            cost_10.append(c10_1)
            cost_10.append(c10_2)

            old_game = currGame.clone()
            new_game = currGame.clone()

            new_game.move(manager.movename(m_number))

            currGame.grid_ = new_game.grid_
            if(currGame != old_game):
                currGame.evolve()
            else:
                done = True

            print(i, r)

            if (i + 1) % 250 == 0:

                print(c8_1, c8_2)
                print(c9_1, c9_2)
                print(c10_1, c10_2)

                print(average(cost_8))
                print(average(cost_9))
                print(average(cost_10))

                print(heighest)


            if (i + 1) % 5000 == 0:


                cost_8 = []
                cost_9 = []
                cost_10 = []

                print("saving")
                neuralnet.saveTrainNeuralNet(net_8_1, sess, ('net_8_1_' + str(i + 1)))
                neuralnet.saveTrainNeuralNet(net_8_2, sess, ('net_8_2_' + str(i + 1)))
                neuralnet.saveTrainNeuralNet(net_9_1, sess, ('net_9_1_' + str(i + 1)))
                neuralnet.saveTrainNeuralNet(net_9_2, sess, ('net_9_2_' + str(i + 1)))
                neuralnet.saveTrainNeuralNet(net_10_1, sess, ('net_10_1_' + str(i + 1)))
                neuralnet.saveTrainNeuralNet(net_10_2, sess, ('net_10_2_' + str(i + 1)))
