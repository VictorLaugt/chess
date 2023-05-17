# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 13:30:59 2022

@author: victor
"""

# nouveaux noyaux
import chess
import fenstrings

# anciens noyaux
import chess_objects_4_2 as old_chess


import time
from typing import Union, Callable, List, Tuple, Dict



def count_actions_1(game:old_chess.ChessGameControler, depth:int) -> int :
    if depth == 0 :
        return 1
    
    count = 0
    for i, line in enumerate(game.tab) :
        for j, piece in enumerate(line) :
            if piece.colour == game.turn :
                for end_square in piece.movements(game, i, j) :
                    game.play_movement((i, j), end_square)
                    count += count_actions_1(game, depth-1)
                    game.play_undo()
    return count


def count_actions_2(game:chess.Chessgame, depth:int) -> int :
    if depth == 0 :
        return 1
    
    count = 0
    for act in game.possibilities() :
        if act.do() :
            count += count_actions_2(game, depth-1)
            act.undo()
    return count


class Chrono :
    __slots__ = ('start', 'stop')
    def __init__(self) :
        self.start : float = None
        self.stop : float = None
    
    def __enter__(self) :
        t = time.time()
        self.start = t
        return self
    
    def __exit__(self, exception_type:type, exception_value:Exception, exception_traceback) :
        t = time.time()
        self.stop = t
        print(self.perf())
        
    def perf(self) :
        if self.start is not None and self.start is not None :
            return self.stop - self.start
        else :
            return None


class BugTrackerFenstrings :
    position1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    position2 = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -"
    position3 = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -"
    position4 = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"
    position5 = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"


if __name__ == '__main__' :
    core = chess.Chessgame()
    old_core = old_chess.ChessGameControler()

