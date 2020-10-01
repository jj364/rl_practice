#! /usr/bin/env python

import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

DECK = {'A':11, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, '10':10, 'J':10, 'Q':10, 'K':10}
DEALER_POLICY = 17  # Fixed policy for dealer. If Ace is useable it MUST be 11


def deal():
    """
    Randomly sample from DECK which is assumed to be infinite and random to ignore ability to 'count cards'
    :return: card key
    """
    return random.sample(DECK.keys(), 1)[0]


def use_ace(hand):
    """
    Convert the Ace with value 11 to ace of value 1
    :param hand: Players hand, will contain 'A'
    :return: updated hand
    """
    aces = np.where(np.array(hand) == 'A')
    hand[aces[0][0]] = 'a'
    return hand


def hand_val(h):
    """
    Calculate value of hand given cards
    :param h: Hand
    :return: Numerical value of hand
    """
    val = 0
    for c in h:
        val += card_val(c)
    return val


def card_val(c):
    """
    Consults the DECK dictionary to determine value
    :param c: Card from hand
    :return:
    """
    if c != 'a':
        return DECK[c]
    else:
        return 1


def select_action(hand, val, policy):
    """
    Given policy and hand it determines whether to hit or stick
    :param hand: Current hand
    :param val: Value of hand
    :param policy: Player/dealer policy used to determine action
    :return: Action and Hand (will be different if Ace 11 turned to ace 1)
    """
    n_aces = np.count_nonzero(np.array(hand) == 'A')  # Calculate if there is useable ace
    if type(policy) is int:
        if int(policy) <= val <= 21:
            return 'stick', hand
        elif val > 21 and n_aces != 0:  # Use Ace if we would go bust
            hand = use_ace(hand)
            return select_action(hand, hand_val(hand), policy), hand
        elif int(policy) <= val:
            return 'stick', hand
        else:
            return 'hit', hand
    elif type(policy) is float:  # Random policy
        return np.random.choice(['hit', 'stick'], p=[policy, 1-policy]), hand


def monte_carlo_off_policy(n_episodes, target_policy, behaviour_policy):
    """
    Play multiple games using 2 policies to determine the value of the target policy
    :param n_episodes: Number of games to be played
    :param target_policy: Fixed, deterministic policy
    :param behaviour_policy: Stochastic policy used to determine value of target
    :return: Ordinary sampling and weighted sampling values
    """
    rhos = np.zeros(n_episodes)
    returns = np.zeros(n_episodes)
    for e in range(n_episodes):
        # Play multiple games
        reward, player_trajectory = game(behaviour_policy, initial_state=[['A','2'],['2']])
        rho = 1.0  # importance sampling ratio between 2 policies

        for (useable_ace, player_hand, player_value, dealer_start_value), action in player_trajectory:
            target_action, _ = select_action(player_hand, player_value, target_policy)
            if action == target_action:
                rho /= 0.5  # Behaviour policy is random so denominator needs to be halved
            else:
                rho = 0.0
                break
        rhos[e] = rho
        returns[e] = reward

    weighted_rtns = rhos * returns
    sum_weighted_rtns = np.cumsum(weighted_rtns)
    sum_rhos = np.cumsum(rhos)

    ordinary_sampling = sum_weighted_rtns/np.arange(1, n_episodes+1)
    weighted_sampling = np.array([sum_weighted_rtns[i]/sum_rhos[i] if rhos[i] != 0 else 0 for i in range(n_episodes)])

    return ordinary_sampling, weighted_sampling


def game(player_policy, initial_state=None):
    """
    Play single game of blackjack
    :param player_policy: Policy which player will used. Int if policy is threshold or Float if stochastic
    :param initial_state: [[PLAYER HAND],[DEALER HAND]]
    :return: reward, player_trajectory. Result of game and hand/actions taken by player during game
    """
    # create player_trajectory variable for analysis
    player_traj = []

    if initial_state:  # Can specify starting hand for testing
        dealer_hand = initial_state[1]
        while len(dealer_hand) != 2:
            dealer_hand.append(deal())
        player_hand = initial_state[0]
    else:
        dealer_hand = [deal(), deal()]  # Deal 2 cards to player and dealer
        player_hand = [deal(), deal()]

    while hand_val(player_hand) < 12:  # Will always hit on less than 12 so get another card
        player_hand.append(deal())

    if hand_val(player_hand) == 22:  # This can only happen in case of A A (use the ace basically)
        _, player_hand = select_action(player_hand, 22, player_policy)

    dealer_start_value = card_val(dealer_hand[0])
    if 'A' in player_hand:
        useable_ace = True
    else:
        useable_ace = False
    dealer_value = hand_val(dealer_hand)
    player_value = hand_val(player_hand)

    while True:  # Player's turn
        if player_value > 21:  # Bust, dealer win
            return -1, player_traj

        action, player_hand = select_action(player_hand, player_value, player_policy)
        player_traj.append([(useable_ace, player_hand, player_value, dealer_start_value), action])

        if action == 'hit':
            player_hand.append(deal())
            player_value = hand_val(player_hand)

        else:
            break

    while True:  # Dealer's turn
        action, dealer_hand = select_action(dealer_hand, dealer_value, DEALER_POLICY)

        if action == 'hit':
            dealer_hand.append(deal())
            dealer_value = hand_val(dealer_hand)
        else:
            break

    if dealer_value > 21 or dealer_value < player_value:
        r = 1  # Player win
    elif dealer_value > player_value:
        r = -1  # Dealer win
    else:
        r = 0  # Draw
    return r, player_traj


# Start game with policy as hit/stick value threshold i.e. '20' means stick on 20 or above
# Possible starting states [12->21], [A-10], useable ace
def run_value_fn():
    state_value_fn = np.zeros((10, 10, 2, 2))   # 3rd dim is useable ace Bool; 4th dim is for reward, count

    for g in range(100000):
        reward, player_trajectory = game(20)
        (ace, ph, pv, dv) = player_trajectory[0][0]
        state_value_fn[pv-12, dv-2, int(ace), 0] += reward
        state_value_fn[pv-12, dv-2, int(ace), 1] += 1

    for row in range(10):
        for col in range(10):
            if state_value_fn[row, col, 0, 1] != 0:
                state_value_fn[row, col, 0, 0] /= state_value_fn[row, col, 0, 1]  # Normalise values
            if state_value_fn[row, col, 1, 1] != 0:
                state_value_fn[row, col, 1, 0] /= state_value_fn[row, col, 1, 1]  # Normalise values

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax2 = fig.add_subplot(211, projection='3d')
    X, Y = np.meshgrid(np.arange(10), np.arange(10))
    ax.plot_wireframe(X, Y, state_value_fn[:,:,0,0])
    ax2.plot_wireframe(X, Y, state_value_fn[:,:,1,0])
    plt.show()


def plot_mc_off_policy():
    target_policy = 20  # Deterministic policy where stick on 20 or 21
    behaviour_policy = 0.5  # Random hit/stick policy

    runs = 10
    eps = 1000
    predicted_val = -0.27726

    ord_samp_errors = np.zeros(eps)
    wght_samp_errors = np.zeros(eps)

    for r in range(runs):
        ordinary_sampling, weighted_sampling = monte_carlo_off_policy(eps, target_policy, behaviour_policy)
        ord_samp_errors += np.power(ordinary_sampling - predicted_val, 2)
        wght_samp_errors += np.power(weighted_sampling - predicted_val, 2)

    ord_samp_errors /= runs  # Average errors over runs
    wght_samp_errors /= runs

    # Compare ordinary and weighted sampling
    fig = plt.figure()
    ax = fig.add_subplot('111')
    ax.set_xscale('symlog')
    ax.plot(ord_samp_errors, label='Ordinary Sampling')
    ax.plot(wght_samp_errors, label='Weighted Sampling')
    ax.set_xlabel('Episodes (log scale)')
    ax.set_ylabel('Mean Squared Error')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_mc_off_policy()
