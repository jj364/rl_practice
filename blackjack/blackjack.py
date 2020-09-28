#! /usr/bin/env python

import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

DECK = {'A':11, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, '10':10, 'J':10, 'Q':10, 'K':10}
DEALER_POLICY = '17'  # Fixed policy for dealer. If Ace is useable it MUST be 11

def deal():
    """
    Randomly sample from DECK which is assumed to be infinite and random to ignore ability to 'count cards'
    :return: card key
    """
    return random.sample(DECK.keys(), 1)[0]


def use_ace(hand):
    aces = np.where(np.array(hand) == 'A')
    hand[aces[0][0]] = 'a'
    return hand


def hand_val(h):
    val = 0
    for c in h:
        val += card_val(c)
    return val


def card_val(c):
    if c != 'a':
        return DECK[c]
    else:
        return 1


def select_action(hand, val, policy):
    n_aces = np.count_nonzero(np.array(hand) == 'A')  # Calculate if there is useable ace
    if int(policy) <= val <= 21:
        return 'stick', hand
    elif val > 21 and n_aces != 0:  # Use Ace if we would go bust
        hand = use_ace(hand)
        return select_action(hand, hand_val(hand), policy), hand
    elif int(policy) <= val:
        return 'stick', hand
    else:
        return 'hit', hand


def game(player_policy, initial_state=None, value_fn=None):
    if initial_state:
        dealer_hand = initial_state[1]
        while len(dealer_hand) != 2:
            dealer_hand.append(deal())
        player_hand = initial_state[0]
    else:
        dealer_hand = [deal(), deal()]  # Deal 2 cards to player and dealer
        player_hand = [deal(), deal()]

    while hand_val(player_hand) < 12:
        player_hand.append(deal())

    if hand_val(player_hand) == 22:
        _, player_hand = select_action(player_hand, 22, player_policy)

    player_start_value = hand_val(player_hand)
    dealer_start_value = card_val(dealer_hand[0])
    dealer_value = hand_val(dealer_hand)
    player_value = hand_val(player_hand)

    # print(f'Player initial hand: {player_hand}, Value: {player_value}')

    while True:  # Player's turn

        action, player_hand = select_action(player_hand, player_value, player_policy)

        if action == 'hit':
            player_hand.append(deal())
            player_value = hand_val(player_hand)
        else:
            break

    # print(f'Player final hand: {player_hand}, Value: {player_value}')

    if player_value > 21:  # Bust, dealer win
        return -1, player_start_value, dealer_start_value

    # print(f'Dealer initial hand: {dealer_hand}, Value: {dealer_value}')
    while True:  # Dealer's turn
        action, dealer_hand = select_action(dealer_hand, dealer_value, DEALER_POLICY)

        if action == 'hit':
            dealer_hand.append(deal())
            dealer_value = hand_val(dealer_hand)
        else:
            break
    # print(f'Dealer final hand: {dealer_hand}, Value: {dealer_value}')

    if dealer_value > 21 or dealer_value < player_value:
        r = 1  # Player win
    elif dealer_value > player_value:
        r = -1  # Dealer win
    else:
        r = 0  # Draw
    return r, player_start_value, dealer_start_value


# Start game with policy as hit/stick value threshold i.e. '20' means stick on 20 or above
# Possible starting states [12->21], [A-10], useable ace -> Not including this
state_value_fn = np.zeros((10, 10, 2))   # 3rd dim is for reward, count

for g in range(100000):
    reward, pv, dv = game('20')
    state_value_fn[pv-12, dv-2, 0] += reward
    state_value_fn[pv-12, dv-2, 1] += 1

print(state_value_fn)

for row in range(10):
    for col in range(10):
        if state_value_fn[row, col, 1] != 0:
            state_value_fn[row, col, 0] /= state_value_fn[row, col, 1]  # Normalise values

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(np.arange(10), np.arange(10))
ax.plot_wireframe(X, Y, state_value_fn[:,:,0])
plt.show()