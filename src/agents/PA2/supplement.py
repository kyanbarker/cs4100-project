#q0
# using 'X', 'O', and '.' for empty square
# ordered 0 - 8 st
# 0 | 1 | 2
# 3 | 4 | 5
# 6 | 7 | 8
# order does matter as X always goes first which means it's X's turn when there are the
# same number of X and O


EMPTY = '.'

#builds state row by row
def makeState(top, mid, bot):
    return top + mid + bot

#q1
LINES = {(0, 1, 2), (3, 4, 5), (6, 7, 8), #rows
         (0, 3, 6), (1, 4, 7), (2, 5, 8), #cols
         (0, 4, 8), (2, 4, 6)} #diags


def utility(state):
    for a, b, c in LINES:
        if state[a] != EMPTY and state[a] == state[b] == state[c]:
            return 1 if state[a] == 'X' else -1
    if EMPTY not in state:
        return 0
    return None


def test_util():
    assert utility(makeState('XXX', 'OO.', '...')) == 1  # x win w/ row
    assert utility(makeState('O.X', 'OX.', 'O.X')) == -1 # o win w/ col
    assert utility(makeState('XOX', 'XOO', 'OXX')) == 0  # draw
    assert utility(makeState('X.O', '.XO', '..X')) == 1  # x win w/ diag
    print('all unit tests passed for utility()')

#q2
def player(state):
    return 'X' if state.count('X') == state.count('O') else 'O'

def actions(state):
    return [square for square in range(9) if state[square] == EMPTY]

def result(state, action):
    return state[:action] + player(state) + state[action + 1:]

def value(state):
    u = utility(state)
    if u is not None:
        return u, None
    if player(state) == 'X':
        return max_value(state)
    return min_value(state)

def max_value(state):
    bestValue = float('-inf')
    bestAction = None
    for action in actions(state):
        successor, _ = value(result(state, action))
        if successor > bestValue:
            bestValue = successor
            bestAction = action
    return bestValue, bestAction

def min_value(state):
    bestValue = float('inf')
    bestAction = None
    for action in actions(state):
        successor, _ = value(result(state, action))
        if successor < bestValue:
            bestValue = successor
            bestAction = action
    return bestValue, bestAction

def minimax(state):
    v, _ = value(state)
    return v

#q3
s0 = makeState('...', '...', '...')
s6 = makeState('OOX', 'X.O', '..X')
SQUARE_NAME =  ['top_left', 'top_mid', 'top_right',
                'mid_left', 'mid_mid', 'mid_right',
                'bot_left', 'bot_mid', 'bot_right']
def verify():
    v, best = value(s6)
    print('minmax(s6), X to move to %d (best move: %s)' % (v, SQUARE_NAME[best]))
    for action in actions(s6):
        print('X plays the %-19s -> minimax = %2d'
              % (SQUARE_NAME[action], minimax(result(s6, action))))
    assert v == 1, 'X should be able to force a win from s6'

    blank = minimax(s0)
    print('minimax(blank board) = %d' % blank)
    assert blank == 0, 'perfect play from the blank board should be a draw'

#q4
s1 = makeState('...', '...', '..X')
s2 = makeState('O..', '...', '..X')
s3 = makeState('O..', 'X..', '..X')
s4 = makeState('OO.', 'X..', '..X')
s5 = makeState('OOX', 'X..', '..X')
GAME = [s0, s1, s2, s3, s4, s5, s6]

def optimal(state):
    target = minimax(state)
    return [action for action in actions(state) if minimax(result(state, action)) == target]

def analyze():
    for k in range(1, len(GAME)):
        s = GAME[k - 1]
        s_next = GAME[k]
        mover = player(s)
        move = next(square for square in range(9) if s[square] != s_next[square])
        before = minimax(s)
        after = minimax(s_next)
        verdict = 'optimal' if after == before else 'suboptimal'
        print('Move %d: %s plays the %-19s minimax(s%d) = %2d -> minimax(s%d) = %2d   %s'
              % (k, mover, SQUARE_NAME[move], k - 1, before, k, after, verdict))
        if after != before:
            alt = [SQUARE_NAME[a] for a in optimal(s)]
            print('optimal alternative: ' + ', '.join(str(a) for a in alt))

"""
Move 2: O plays the top_left            minimax(s1) =  0 -> minimax(s2) =  1   suboptimal
optimal alternative: mid_mid
Move 3: X plays the mid_left            minimax(s2) =  1 -> minimax(s3) =  0   suboptimal
optimal alternative: top_right, bot_left
Move 4: O plays the top_mid             minimax(s3) =  0 -> minimax(s4) =  1   suboptimal
optimal alternative: top_right, mid_mid, mid_right
"""



if __name__ == '__main__':
    test_util()
    print()
    verify()
    print()
    analyze()