import sys
from random import randint, random
import numpy as np
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

class Game2048(object):
    """
    Abstraction of the 2048 game.
    Paramters
    ---------
    size : int in [2, 4, 8], optional (default None)
        Size of game board.
    grid : 2D array of shape (size, size), optional (default None)
        Values to initialize game board (useful for cloning game instances)
    mode : string in ["arrows", "letters"], optional (default "arrows")
        Specifies how input is presented to the game.
    mover : callable, optional (default None)
        Returns next move to be made. If none, next moves will asked at the
        stdin as game play proceeds.
    """

    def __init__(self, size=None, grid=None, random_state=None, mode='arrows',
                 mover=None):
        # misc
        if not mode in ["arrows", "letters"]:
            raise ValueError("Invalid mode: %s" % mode)
        self.mode = mode
        self.size = size
        self.grid = grid
        self.mover = mover
        self.random_state = random_state
        self.banner_printed_ = False
        self.aborted_ = False
        self._load_batteries()


    def _load_batteries(self):
        """Some serious conf business."""
        self.rng_ = np.random.RandomState(self.random_state)

        # get size of grid
        if self.size is None:
            if not self.grid is None:
                self.size = len(self.grid)
            self.print_banner()
            while not self.size in SIZES:
                try:
                    self.size = int(INPUT_GRABBER(
                        ("<> Enter game size (can be %s):"
                         " ") % ", ".join([str(s) for s in SIZES])))
                except ValueError:
                    continue

        # make grid
        if self.grid is None:
            self.grid_ = np.zeros((self.size, self.size), dtype=int)
            indices = list(np.ndindex((self.size, self.size)))
            support = self.rng_.choice(range(len(indices)), 2)
            self.grid_[indices[support[0]][0], indices[support[0]][1]] = 2
            self.grid_[indices[support[0]][1],
                       indices[support[1]][1]] = self.rng_.choice([2, 4])
        else:
            self.grid_ = np.array(self.grid, dtype=int).copy()

    def get_params(self):
        """Get all parameters of class instance."""
        return dict(size=self.size, grid=self.grid_.copy(),
                    random_state=self.random_state)

    def clone(self):
        """Clone class instance."""
        return Game2048(**self.get_params())

    def __eq__(self, other):
        """Compare game with another instance."""
        return np.all(self.grid_ == other.grid_)

    def __ne__(self, other):
        """Compare game with another instance."""
        return np.any(self.grid_ != other.grid_)

    def get_move(self):
        """Get next move from input sensor (screen, etc.)."""
        if not self.mover is None:
            mv = self.mover(self.clone())
            print("<> Performing %s" % [k for k, v in MOVES.items()
                                        if v == mv][0])
        elif self.mode == "arrows":
            mv = GetArrow()()
            if not mv is None:
                mv = MOVES[mv]
        else:
            while not mv in ["I", "J", "K", "L"]:
                mv = INPUT_GRABBER(("<> Enter direction for movement "
                                    "(I=u,J=left,K=down, L=right) : "))

                # check empty line
                if not mv:
                    mv = None
        return mv

    def __repr__(self):
        """Converts "2048" board to a string."""
        out = ""
        line = None
        for line in [[(str(x) if x else "").center(6) + "|" for x in line]
                     for line in self.grid_]:
            line = "|%s" % "".join(line)
            out += "%s\n" % line
            out += "%s\n" % ("-" * len(line))
        out = "%s\n%s" % ("-" * len(line), out)
        return out

    def evolve(self):
        """Fill a random empty cell with 2 (or 4, with very small proba)."""
        z = np.where(self.grid_ == 0.)
        i = self.rng_.choice(len(z[0]))
        self.grid_[z[0][i], z[1][i]] = self.rng_.choice([2, 4], p=[.9, .1])
        return self

    def _squeeze(self, v, down=True):
        """Reorganize vector into a contiguous chunk of nonzero stuff, padded
        with zeros."""
        nz = [x for x in v if x]
        if down:
            return np.append(np.zeros(len(v) - len(nz)), nz)
        else:
            return np.append(nz, np.zeros(len(v) - len(nz)))

    def _vertical_move(self, down=True):
        """Helper method for doing up and down moves."""
        places = range(self.size)
        places = places[:0:-1] if down else places[:-1]
        for j in range(self.size):
            col = self._squeeze(self.grid_[:, j], down=down)
            for i in places:
                salt = 1 - 2 * down
                if col[i] == col[i + salt]:
                    # merge cells (i, j) and (i + salt, j)
                    col[i] *= 2
                    col[i + salt] = 0
                    col = self._squeeze(col, down=down)
            self.grid_[:, j] = col
        return self

    def up(self):
        """Perform upward move."""
        return self._vertical_move(down=False)

    def down(self):
        """Perform downward move."""
        return self._vertical_move(down=True)

    def _horizontal_move(self, right=True):
        """Helper method for doing left and right moves.
        The trick is to simply do a horizontal movement on the transpose
        game."""
        params = self.get_params()
        params["grid"] = self.grid_.T
        self.grid_ = Game2048(**params)._vertical_move(down=right).grid_.T
        return self

    def left(self):
        """Perform leftward move."""
        return self._horizontal_move(right=False)

    def right(self):
        """Perform rightward move."""
        return self._horizontal_move(right=True)

    def move(self, mv):
        """Make specified move."""
        if mv == "u":
            return self.up()
        elif mv == "l":
            return self.left()
        elif mv == "d":
            return self.down()
        elif mv == "r":
            return self.right()
        else:
            raise ValueError("Invalid move: %s" % mv)

    @property
    def score(self):
        """Get running score."""
        return np.max(self.grid_)

    @property
    def full(self):
        """Check whether grid is full."""
        return np.all(self.grid_ != 0.)

    @property
    def ended(self):
        """Check whether game has ended."""
        if self.aborted_ or self.score >= 2048:
            return True
        else:
            # dead-end detection
            for mv in MOVES.values():
                game = self.clone().move(mv)
                if game != self:
                    return False
            return True

    def play(self):
        """Play game."""

        while not self.ended:
            print(self)
            while True:
                mv = self.get_move()
                old_game = self.clone()
                game = self.clone()
                if mv is None:
                    print("Game aborted by user")
                    game.aborted_ = True
                    break
                game.move(mv)
                if game != old_game:
                    break
            self.grid_ = game.grid_
            if self.ended:
                break
            self.evolve()

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

#print "ciao"
