#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 10:14:03 2023

@author: victor
"""

import os
import random
from typing import Tuple

import chess
import fenstrings

from debug_ui import start_debug_ui


SEP =  '-'*50


def square_indexes(sqr_id:int) -> Tuple[int]:
    """square_indexes(sqr_id) == couple d'indices (ligne i, colonne j) de la
    case dont l'identifiant est sqr_id dans le noyau
    """
    i = sqr_id//10
    j = sqr_id%10
    return (i, j)


def square_name(sqr_id:int) -> str:
    """square_name(sqr_id) == nom de la case dont l'identifiant est sqr_id dans
    le noyau
    """
    i, j = square_indexes(sqr_id)
    return 'abcdefgh'[j] + str(8-i)


def action_repr(action:chess.Action) -> str:
    """Renvoie au format "pièce (case de départ) -> (case d'arrivée)" la chaine
    de caractère qui représente l'action reçue en paramètre
    """
    piece = action.piece
    start, end = piece.square, action.target_square()
    return (
        f"{piece} ({square_name(start)}->{square_name(end)}) "
        f"{type(action).__name__}"
        )


def print_state(action:chess.Action, game:chess.Chessgame):
    """Affiche l'état actuel de l'échiquier."""
    print(SEP)
    print(action, end='\n\n')
    print(game, end='\n\n\n')


def do_not_print(action:chess.Action, game:chess.Chessgame):
    pass


def play_randomly(game:chess.Chessgame, print_func) -> bool:
    """Si cela est possible, joue une action aléatoire sur l'échiquier `game`
    et renvoie True, sinon, renvoie False.
    """
    possibilities = game.possibilities()
    while possibilities:
        action = possibilities.pop(random.randrange(len(possibilities)))
        if action.do():
            print_func(action, game)
            return True
    return False


def random_player(game:chess.Chessgame, n:int, print_function=do_not_print):
    """
    Joue aléatoirement pendant au plus `n` actions sur l'échiquier `game` et
    revoie la fenstring décrivant l'état final de l'échiquier
    """
    for i in range(n):
        if not play_randomly(game, print_function):
            break
    return fenstrings.create(game)



def display_random_game(game, n, /, *, rng_seed=None, print_function=do_not_print, dbg:bool=False):
    rng_seed = rng_seed or os.urandom(20)
    random.seed(rng_seed)
    print(f"Rng seed used: {rng_seed}")

    final_fenstring = random_player(game, n, print_function)
    print(f"Final state: {final_fenstring}\n\n{game}")

    print("\npossibilities:")
    for action in game.possibilities():
        print(f"    {action_repr(action)}")

    if dbg:
        start_debug_ui(game)


if __name__ == '__main__':
    game = chess.Chessgame()
    def step_by_step(*args, **kwargs):
        input()
        return print_state(*args, **kwargs)

    display_random_game(game, 300, rng_seed=10, print_function=step_by_step)



