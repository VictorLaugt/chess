#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 20:08:42 2022

@author: victor
"""

"""
Construction des arbres par induction :
    B = {Sentinelle}
    K = {(valeur, noeuds) ---> Node(valeur, noeuds)}
"""

import chess
import fenstrings

import actions_gen

import abc

from typing import TypeVar, Union, Tuple, List, Iterator, Callable

Self = TypeVar('Self')

# plateau de jeu utilisé pour calculer la représentation des états contenus dans
# les noeuds des arbres
_GAME_REPR = chess.Chessgame()

class Tree(abc.ABC) :
    """Implémente des arbres d'états"""
    __slots__ = ('fenstring', 'action')

    def __init__(self, fenstring:str, action:chess.Action) :
        self.fenstring = fenstring
        self.action = action

    def board_repr(self) -> str:
        fenstrings.load(_GAME_REPR, self.fenstring)
        return _GAME_REPR._board_repr()

    def display_board(self):
        print(self.board_repr())

    def depth(self) -> int :
        """Renvoie la profondeur de l'arbre"""

    @abc.abstractmethod
    def __getitem__(self, ind:Union[int, Tuple[int]]) -> Self :
        """tree[i, j, k, ...] == tree.linked[i].linked[j].linked[k] ...
        """

    @abc.abstractmethod
    def branches(self, leaf_condition:Callable) :
        """Renvoie toutes les branches dont la feuille vérifie la confition
        leaf_condition
        """

Branch = List[Tree]


class Leaf(Tree) :
    """Implémente les feuilles des arbres d'états

    fenstring : str
        fenstring qui décrit l'état représenté par cette feuille

    action : Union[chess.Action, None]
        action qui a permit d'atteindre cet état
    """
    __slots__ = ()

    def __repr__(self) -> str :
        return f"<{type(self).__name__} at {hex(id(self))}>"

    def depth(self) -> int :
        return 0

    def __getitem__(self, indexes) :
        if isinstance(indexes, (int, tuple)) :
            raise IndexError
        raise TypeError

    def branches(self, leaf_condition:Callable, path) -> Iterator[Branch] :
        if leaf_condition(self) :
            yield path + [self]


class Node(Tree) :
    """Implémente les noeuds des arbres d'états

    fenstring : str
        fenstring qui décrit l'état représenté par cette feuille

    action : Union[Chess.Action, None]
        action qui a permit d'atteindre cet état

    linked : List[Tree]
        états atteignables depuis celui-ci
    """
    __slots__ = ('linked', )

    def __init__(self, fenstring:str, action:Union[chess.Action, None]) :
        super().__init__(fenstring, action)
        self.linked = []

    def __repr__(self) -> str :
        return (
            f"<{type(self).__name__} at {hex(id(self))} "
            f"(linked to {len(self.linked)} other states)>"
        )

    def depth(self) -> int :
        d = 0
        state = self
        while not isinstance(state, Leaf) :
            state = state.linked[0]
            d += 1
        return d

    def __getitem__(self, ind:Union[int, Tuple[int]]) :
        if isinstance(ind, int) :
            if ind < len(self.linked) :
                return self.linked[ind]
            raise IndexError

        elif isinstance(ind, tuple) :
            state = self
            for i in ind :
                state = state[i]
            return state

        raise TypeError


    def branches(self, leaf_condition:Callable, path) -> Iterator[Branch] :
        path = path + [self]
        for state in self.linked :
            yield from state.branches(leaf_condition, path)


def _state_tree_builder(game : chess.Chessgame,
                        depth : int,
                        done : chess.Action) -> Tree :
    fen = fenstrings.create(game)

    if depth == 0 :
        return Leaf(fen, done)

    node = Node(fen, done)
    for act in game.possibilities() :
        actions_gen.save_representation(act)
        if act.do() :
            node.linked.append( _state_tree_builder(game, depth-1, act) )
            act.undo()

    return node


def state_tree(game : chess.Chessgame, depth:int) -> Tree :
    return _state_tree_builder(game, depth, None)



if __name__ == '__main__' :
    FENSTRING = actions_gen.BugTrackerFenstrings.position2
    g = chess.Chessgame()
    fenstrings.load(g, FENSTRING)



