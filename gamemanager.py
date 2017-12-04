import sys
from random import randint, random
import copy, time

class _Getch:
    """Gets a single character from standard input.  Does not echo to the screen."""
    def __init__(self):
        try:
            self.impl = _GetchWindows()
        except ImportError:
            self.impl = _GetchUnix()

    def __call__(self): return self.impl()

class _GetchUnix:
    def __init__(self):
        import tty, sys

    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

class _GetchWindows:
    def __init__(self):
        import msvcrt

    def __call__(self):
        import msvcrt
        return msvcrt.getch()

class Map:
    data = None
    height = None
    width = None

    def __init__(self, h, w, start_count, copy_from=None):
        if copy_from:
            self.height = copy_from.height
            self.width = copy_from.width
            self.data = [row[:] for row in copy_from.data]
            self.movements = copy_from.movements
            return

        if h == 0 or w == 0:
            raise Exception("Wrong dimensions.")

        self.height = h
        self.width = w

        self.movements = self.generate_movements()

        self.data = [[0 for i in range(self.width)] for j in range(self.height)]
        for i in range(min(start_count, self.height*self.width)):
            x, y = self.get_random_empty()
            self.data[x][y] = 1

    def get_random_empty(self):
        squares = self.height*self.width
        start_n = n = randint(0, squares-1)
        k = 0
        while True:
            i = int(k/self.width)
            j = k%self.width
            if self.data[i][j] == 0:
                if n == 0:
                    return [i, j]
                n -= 1
            k +=1
            if k == squares:
                if n == start_n: #no empty
                    return None
                k = 0

    def set_cell(self, x, y, value):
        self.data[x][y] = value

    def get_empty_cells(self):
        cells = []
        for i in range(self.height):
            for j in range(self.width):
                if self.data[i][j] == 0:
                    cells.append([i, j])
        return cells

    def print_map(self, nl=True):
        print ("+-----"*self.width, "+")
        for i in range(self.height):
            d = ""
            for j in range(self.width):
                if self.data[i][j] == 0:
                    d += "|    "
                else:
                    d += "|%4d"%(self.data[i][j])
            print(d)
            print ("|")
            print ("+-----"*self.width + "+")
        if nl:
            print ("")

    def is_valid(self, x, y):
        return x >= 0 and x < self.height and y >= 0 and y < self.width

    def generate_movements(self):
        return {'l': [[x, y] for x in range(0, self.height) for y in range(0, self.width, 1)],
                'r': [[x, y] for x in range(0, self.height) for y in range(self.width-1, -1, -1)],
                'd': [[y, x] for x in range(0, self.width) for y in range(self.height-1, -1, -1)],
                'u': [[y, x] for x in range(0, self.width) for y in range(0, self.height, 1)]}

    def move(self, direction='l'):
        di, dj = {'l': [0, -1], 'r': [0, +1], 'd': [1, 0], 'u':[-1, 0]}[direction]
        mov = self.movements[direction]

        merged = {}
        m = self.data
        t = [[m[x][y] for y in range(len(m[0]))] for x in range(len(m))]

        for i, j in mov:
            while self.is_valid(i+di, j+dj) and m[i+di][j+dj] == 0:
                merged[(i+di, j+dj)] = True
                m[i+di][j+dj] = m[i][j]
                m[i][j] = 0
                j += dj
                i += di
            if self.is_valid(i+di, j+dj) and m[i+di][j+dj] == m[i][j] and \
                    not merged.get((i+di, j+dj)):
                merged[(i+di, j+dj)] = True
                m[i+di][j+dj] += 1
                m[i][j] = 0

        if( t == m):
            print("illegal")
            return -1
        self.data = m
        return 1


    def get_copy(self):
        return Map(0, 0, 0, copy_from=self)

    def equal(self, map):
        if self.height != map.height or self.width != map.width:
            return False

        for i in range(self.height):
            for j in range(self.width):
                if self.data[i][j] != map.data[i][j]:
                    return False

        return True

def new_cell_value():
    if random() > 0.1:
        return 1
    else:
        return 2

def play(height=4, width=4, init=2):
    getch = _Getch()
    keys = {"j":"d", "h":"l", "k":"r", "u":"u"}
    print( "Press h for left, k for right, j for down and u for up.")
    m = Map(height, width, init)
    while True:
        m.print_map()
        char = getch()
        if char == "q":
            break
        if char not in keys:
            print ("Unknown char.")
            continue
        m.move(keys[char])

        p = m.get_random_empty()
        if p:
            m.set_cell(p[0], p[1], new_cell_value())
        else:
            print ("You loose")
            break


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


def get_best_move(m):
    return find_best_move(m)

def movename(move):
    return ['u', 'd', 'l', 'r'][move]

if __name__ == "__main__":
    play(4, 4)

#print "ciao"
