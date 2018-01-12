import tensorflow as tf
import gamemanager as gm
import neuralnet as nn
import numpy as np

def splitboard(board):
    return board.tolist()

def transpose_board(board, heighest):
    b2 = []

    for x in range(len(board)):
        for y in range(len(board[x])):
            b2.append(float(board[x][y] / heighest))

    return [b2]

def get_move_percentile(softmax):
    ran_float = np.random.random(1)

    i = 0
    total = (softmax[0][0] / 100)
    running = True

    while running:

        if ran_float < total or i == (len(softmax[0]) - 1):
            return i

        i += 1
        total += (softmax[0][i] / 100)

    return 0

def get_move(softmax):
    heighest = 0
    index = 0
    for i in range(4):
        if softmax[0][i] > heighest:
            heighest = softmax[0][i]
            index = i

    return index

def find_heighest(board):
    heighest = 0
    for x in board:
        for y in x:
            if(y > heighest):
                heighest = y
    return heighest

def score_board(board, heighest):
    score = 0

    for r in board:
        for v in r:
            score += v

    return score + heighest

def check_board(game):
    same = True
    for i in range(4):
        old_game = game.clone()
        new_game = game.clone()

        new_game.move(gm.movename(i))
        if(new_game != old_game):
            return (False, i)

    return True, 0

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

def average(numbers):
    total = 0
    for i in numbers:
        total += i
    total = total/len(numbers)
    return total

def std_dev(numbers, average):
    s = 0
    for n in numbers:
        s += (n - average)*(n - average)

    s = s / len(numbers)
    s = np.sqrt(s)

    return s

def play(neuralnet, sess):
    game = gm.Game2048(4)
    done = False

    ill = 0
    i = 0
    while not done:
        data = splitboard(game.grid_)
        #data = log2board(data)
        heighest = find_heighest(data)
        board = transpose_board(data, heighest)

        softmax = sess.run(neuralnet.softmax_output, feed_dict={neuralnet.input_placeholder:  board})

        m = get_move(softmax)

        #if i % 100 == 0:
            #print(find_heighest(board))

        old_game = game.clone()
        new_game = game.clone()

        if(ill == 10):
            same, move = check_board(new_game)
            if(same):
                return new_game.grid_

            new_game.move(gm.movename(move))
            ill = 0
        else:
            new_game.move(gm.movename(m))

        game.grid_ = new_game.grid_

        if(game != old_game):
            game.evolve()
        else:
            ill += 1

        i += 1

def load_net(structure, iteration):
    d = []
    filename = "net_" + str(structure) +  "_1_" + str(iteration)
    d.append(nn.loadNeuralNet(filename))
    filename = "net_" + str(structure) +  "_2_" + str(iteration)
    d.append(nn.loadNeuralNet(filename))
    return d

def save_results(name, net, boards, results, heighest):
    with open(("results/" + name), 'w') as f:
        f.write('strucuture: \n')
        for s in net.structure:
            f.write(str(s) + " ")

        av = average(results)
        print(av)
        sd = std_dev(results, av)
        f.write('\n')
        f.write("Average score: " + str(av))

        f.write('\n')
        f.write("std_dev: " + str(sd))

        av = average(heighest)
        print(av)
        sd = std_dev(heighest, av)
        f.write('\n')
        f.write("Average score: " + str(av))

        f.write('\n')
        f.write("std_dev: " + str(sd))



        i = 0
        for i in range(len(results)):
            if i % 10 == 0:
                f.write('\n')
            f.write(str(results[i]) + ' ' + str(heighest[i]) + ' ')


#        f.write('\n')
#        for b in boards:
#            f.write(str(find_heighest(b)) + '\n')
#
#            for r in b:
#                for v in r:
#                    f.write(str(r) + " ")
#                f.write('\n')
#            f.write('\n')

def play_nets(name, episodes, it, sess):
    nets = []
    nets.append(load_net(8, it))
    nets.append(load_net(9, it))
    nets.append(load_net(10, it))

    sess.run(tf.global_variables_initializer())

    n = 0
    for net_2 in nets:
        for net in net_2:
            boards = []
            scores = []
            heighest = []

            for i in range(episodes):
                if i % 50 == 0:
                    print("At 50 ")
                boards.append(play(net, sess))
                heighest.append(find_heighest(boards[i]))
                scores.append(score_board(boards[i], heighest[i]))


            save_results(name + "Net_" + str(n), net, boards, scores, heighest)
            n += 1





score = []
n_move = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    episodes = 100
#    play_nets("IT_050K", episodes, 50000,  sess)
#    play_nets("IT_100k", episodes, 100000, sess)
#    play_nets("IT_150k", episodes, 150000, sess)
#    play_nets("IT_200k", episodes, 200000, sess)
#    play_nets("IT_250k", episodes, 250000, sess)
    play_nets("IT_5k", episodes, 5000, sess)
