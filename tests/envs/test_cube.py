from deepspeedcube.envs import get_env

import numpy as np


env = get_env("cube")

""" Moves are tested in two phases. First, env.move
is tested. Then, it is tested that env.multiple_moves
always returns the same as env.move. """

def test_reverse_move():
    # Test that applying moves and then reverse moves always
    # ends up in the original state
    for _ in range(10):
        state = np.random.randint(0, 24, 20, dtype=np.uint8)
        orig_state = state.copy()
        actions = [np.random.choice(env.action_space) for j in range(100)]
        reverse_actions = [env.reverse_move(action) for action in actions][::-1]
        for action in actions:
            state = env.move(action, state)
        for action in reverse_actions:
            state = env.move(action, state)
        assert np.all(state == orig_state)
        assert state.dtype == orig_state.dtype

def test_reverse_moves():
    actions = np.array([np.random.choice(env.action_space) for _ in range(10)], dtype=np.uint8)
    reverse_actions = np.array([env.reverse_move(a) for a in actions], dtype=np.uint8)
    reverse_actions2 = env.reverse_moves(actions)
    assert np.all(reverse_actions == reverse_actions2)
    assert reverse_actions.dtype == reverse_actions2.dtype

def test_move():
    # Specific test case that moves are correct
    state = env.get_solved()
    for action in env.action_space:
        state = env.move(action, state)
    # Test that stringify and by extension _as633 works on solved state
    state = env.get_solved()
    assert env.string(state) == "\n".join([
        "      2 2 2            ",
        "      2 2 2            ",
        "      2 2 2            ",
        "4 4 4 0 0 0 5 5 5 1 1 1",
        "4 4 4 0 0 0 5 5 5 1 1 1",
        "4 4 4 0 0 0 5 5 5 1 1 1",
        "      3 3 3            ",
        "      3 3 3            ",
        "      3 3 3            ",
    ])
    # Perform moves and check if states are assembled/not assembled as expected
    moves = (6, 0, 6, 7, 2, 3, 9, 8, 1, 0)
    assembled = (False, True, False, False, False, False, False, False, False, True)
    for m, a in zip(moves, assembled):
        state = env.move(m, state)
        assert a == env.is_solved(state)

    # Perform move and check if it fits with how the string representation would look
    state = env.get_solved()
    state = env.move(m, state)
    assert env.string(state) == "\n".join([
        "      2 2 2            ",
        "      2 2 2            ",
        "      4 4 4            ",
        "4 4 3 0 0 0 2 5 5 1 1 1",
        "4 4 3 0 0 0 2 5 5 1 1 1",
        "4 4 3 0 0 0 2 5 5 1 1 1",
        "      5 5 5            ",
        "      3 3 3            ",
        "      3 3 3            ",
    ])

    # Performs all moves and checks if result fits with how it theoretically should look
    state = env.get_solved()
    moves = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    assembled = (False, False, False, False, False, False,
                 False, False, False, False, False, False)
    for m, a in zip(moves, assembled):
        state = env.move(m, state)
        assert a == env.is_solved(state)
    assert env.string(state) == "\n".join([
        "      2 0 2            ",
        "      5 2 4            ",
        "      2 1 2            ",
        "4 2 4 0 2 0 5 2 5 1 2 1",
        "4 4 4 0 0 0 5 5 5 1 1 1",
        "4 3 4 0 3 0 5 3 5 1 3 1",
        "      3 1 3            ",
        "      5 3 4            ",
        "      3 0 3            ",
    ])

def test_multiple_moves():
    states = np.vstack([env.get_solved()]*len(env.action_space))
    for i, action in enumerate(env.action_space):
        states[i] = env.move(action, states[i])

    states2 = np.vstack([env.get_solved()]*len(env.action_space))
    states2 = env.multiple_moves(env.action_space, states2)
    assert np.all(states == states2)
