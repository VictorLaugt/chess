#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 11:53:54 2022

@author: victor
"""
__all__ = ['explore', 'explore_step_by_step']


from .. import chess
from .. import fenstrings

from time import perf_counter

import abc
from dataclasses import dataclass, field
import colorama

from typing import TypeVar, Union, List, Tuple, Callable, Iterator


Self = TypeVar('Self')
TextColor = TypeVar('TextColor')


# ---- representations
YELLOW_HIGHLIGHT : TextColor = (colorama.Back.YELLOW + colorama.Fore.BLACK +
                                colorama.Style.BRIGHT)
BLUE_HIGHLIGHT : TextColor = (colorama.Back.BLUE + colorama.Fore.RED +
                              colorama.Style.BRIGHT)
RED_HIGHLIGHT = TextColor = (colorama.Back.RED + colorama.Fore.WHITE +
                             colorama.Style.BRIGHT)
REGULAR_COLOR : TextColor = colorama.Style.RESET_ALL

def highlighted(txt:str, highlight_color:TextColor) -> str :
    return f'{highlight_color}{txt}{REGULAR_COLOR}'


def square_name(sqr_id:int) -> str :
    """square_name(sqr_id) == nom de la case dont l'identifiant est sqr_id dans
    le noyau
    """
    try :
        return 'abcdefgh'[sqr_id%10] + str(8-(sqr_id//10))
    except IndexError :
        raise IndexError(sqr_id)


def save_representation(a:chess.Action) :
    """Si a est une action qui n'a pas encore été effectuée sur son échiquier
    ou qui vient d'être annullée, alors sauvegarde une représentation graphique
    de l'action dans son attribut ._debug_repr
    """
    assert (a not in a.board.history)

    try :
        piece = a.piece
        start, end = piece.square, a.target_square()
        action_repr = (
            f"{piece} ({square_name(start)}->{square_name(end)}) ({start}->{end}) "
            f"{type(a).__name__}"
            )
        a._debug_repr = action_repr
    except IndexError :
        print(piece, type(a).__name__, f"{start=}", f"{end=}", a.board,
              sep='\n', end='\n'*2)
        raise


def cat(txt1:str, txt2:str, fill:int=None, sep:str=' | ') -> str :
    result = []
    lines1 = txt1.splitlines()
    lines2 = txt2.splitlines()
    for i, (l1, l2) in enumerate(zip(lines1, lines2)) :
        result.append(l1 + sep + l2)
    if len(lines1) > i+1 :
        result.extend(l1 + sep for l1 in lines1[i+1:])
    elif len(lines2) > i+1 :
        fill = fill or len(lines1[0])
        result.extend(fill*' ' + sep + l2 for l2 in lines2[i+1:])
    return '\n'.join(result)


def board_unicode_repr(game : chess.Chessgame,
                       highlights : Tuple[int],
                       second_highlights : Tuple[int],
                       highlight_check : bool) -> str :
    check_highlights = []
    for colour in (chess.WHITE, chess.BLACK) :
        threat = game.check(colour)
        if threat is not None :
            check_highlights.append(threat.square)
            check_highlights.append(game.kings[colour].square)
    b_repr = ['   A|B|C|D|E|F|G|H|        ']
    for i, num in zip(range(0, 80, 10), range(8, 0, -1)) :
        line = game.array[i:i+8]

        line_repr = []
        for sqr_id, item in enumerate(line, start=i) :
            if sqr_id in highlights :
                line_repr.append(highlighted(item.unicode_repr(), YELLOW_HIGHLIGHT))
            elif sqr_id in second_highlights :
                line_repr.append(highlighted(item.unicode_repr(), BLUE_HIGHLIGHT))
            elif sqr_id in check_highlights :
                line_repr.append(highlighted(item.unicode_repr(), RED_HIGHLIGHT))
            else :
                line_repr.append(item.unicode_repr())

        b_repr.append(f"{num} |{'|'.join(line_repr)}| ({i:02d}-{i+7:02d})")
    return '\n'.join(b_repr)


def board_repr(game : chess.Chessgame,
               highlights : Tuple[int],
               second_highlights : Tuple[int],
               highlight_check : bool) -> str :
    check_highlights = []
    for colour in (chess.WHITE, chess.BLACK) :
        threat = game.check(colour)
        if threat is not None :
            check_highlights.append(threat.square)
            check_highlights.append(game.kings[colour].square)
    b_repr = ['   A |B |C |D |E |F |G |H |        ']
    for i, num in zip(range(0, 80, 10), range(8, 0, -1)) :
        line = game.array[i:i+8]

        line_repr = []
        for sqr_id, item in enumerate(line, start=i) :
            if sqr_id in highlights :
                line_repr.append(highlighted(str(item), YELLOW_HIGHLIGHT))
            elif sqr_id in second_highlights :
                line_repr.append(highlighted(str(item), BLUE_HIGHLIGHT))
            elif sqr_id in check_highlights :
                line_repr.append(highlighted(str(item), RED_HIGHLIGHT))
            else :
                line_repr.append(str(item))

        b_repr.append(f"{num} |{'|'.join(line_repr)}| ({i:02d}-{i+7:02d})")
    return '\n'.join(b_repr)


def state_repr(game:chess.Chessgame,
               highlights : Tuple[int] = (),
               second_highlights : Tuple[int] = (), *,
               highlight_check : bool = True,
               state_variables : bool = False) -> str :
    board = board_repr(game, highlights, second_highlights, highlight_check)
    if state_variables :
        return cat(board, game._info_repr())
    return board


def state_repr_from_action(fen : str,
                           action : chess.Action, *,
                           highlight_check : bool = True,
                           state_variables : bool = False) -> str :
    if action is None :
        return state_repr(
            fenstrings.loaded(fen),
            highlight_check=highlight_check,
            state_variables=state_variables
            )

    elif isinstance(action, chess.Castle) :
        king_start, rook_start = action.piece.square, action.rook.square
        king_end = king_start + action.king_vector
        rook_end = rook_start + action.rook_vector
        return state_repr(
            fenstrings.loaded(fen),
            (king_start, king_end),
            (rook_start, rook_end),
            highlight_check=highlight_check,
            state_variables=state_variables
            )
    else :
        start, end = chess.start_square(action), chess.end_square(action)
        return state_repr(
            fenstrings.loaded(fen),
            (start, end),
            highlight_check=highlight_check,
            state_variables=state_variables
            )


def branch_repr(branch : list, *,
                highlight_check : bool = False,
                state_variables : bool = False) -> str :
    items = [state_repr(fenstrings.loaded(branch[0].state), state_variables=state_variables)]
    for node in branch[1:] :
        state = state_repr_from_action(
            node.state,
            node.action,
            highlight_check=highlight_check,
            state_variables=state_variables
            )
        items.append(f"==>{node.action}==>\n\n{state}")
    return '\n\n'.join(items)


# ---- datatypes
class StateChange :
    def __init__(self, action:chess.Action) :
        self.action : chess.Action = action
        self.before : str = None
        self.after : str = None

    def __repr__(self) :
        return f"StateChange({self.action})"

    def apply(self) -> bool :
        board = self.action.board
        self.before = fenstrings.create(board)
        if self.action.do() :
            self.after = fenstrings.create(board)
            return True
        return False

    def unapply(self) :
        self.action.undo()

    def display(self) :
        action, before, after = self.action, self.before, self.after
        if isinstance(action, chess.Castle) :
            king_start = action.piece.square
            before_repr = state_repr(fenstrings.loaded(before), (king_start,))
        else :
            start = chess.start_square(action)
            before_repr = state_repr(fenstrings.loaded(before), (start,))
        after_repr = state_repr_from_action(after, action)
        print(
            f"{action} :\n\n({before})\n{before_repr}\n\n==>\n\n"
            f"({after})\n{after_repr}"
            )

ActList = Union[List[chess.Action], List[StateChange]]


class ByTags(abc.ABC) :
    def update(self, other) :
        self.action += other.action
        self.capture += other.capture
        self.en_passant += other.en_passant
        self.castle += other.castle
        self.promotion += other.promotion
        self.long_push += other.long_push

    @abc.abstractmethod
    def init_from_action(cls, a:chess.Action) -> Self :
        pass


@dataclass
class CountActions(ByTags) :
    """explore(game, n, CountActions) compte les actions possibles à effectuer
    sur la partie game au niveau de profondeur n.
    """

    action : int = 0 # compte les actions de tout type
    capture : int = 0 # compte les actions capturant une pièce
    en_passant : int = 0 # compte les prises en passant
    castle : int = 0 # compte les roques
    promotion : int = 0 # compte les promotions
    long_push : int = 0 # compte les long pushes

    @classmethod
    def init_from_action(cls, a:chess.Action) -> Self :
        is_en_passant = isinstance(a, chess.EnPassant)
        return cls(
            action = 1,
            capture = (is_en_passant
                       or bool(isinstance(a, chess.Movement) and a.taken)),
            en_passant = is_en_passant,
            castle = isinstance(a, chess.Castle),
            promotion = isinstance(a, chess.Promotion),
            long_push = isinstance(a, chess.LongPush)
            )



@dataclass
class ListActions(ByTags) :
    """explore(game, n, ListActions) liste les actions possibles à effectuer
    sur la partie game au niveau de profondeur n.
    """
    action : ActList = field(default_factory=list) # liste les actions de tout type
    capture : ActList = field(default_factory=list) # liste les actions capturant une pièce
    en_passant : ActList = field(default_factory=list) # liste les prises en passant
    castle : ActList = field(default_factory=list) # liste les roques
    promotion : ActList = field(default_factory=list) # liste les promotions
    long_push : ActList = field(default_factory=list) # liste les long pushes

    @classmethod
    def init_from_action(cls, a:chess.Action) -> Self :
        is_en_passant = isinstance(a, chess.EnPassant)
        return cls(
            action = [a],
            capture = ( [a]
                        if (is_en_passant or bool(isinstance(a, chess.Movement)
                                                  and a.taken))
                        else [] ),
            en_passant = ( [a] if is_en_passant else [] ),
            castle = ( [a] if isinstance(a, chess.Castle) else [] ),
            promotion = ( [a] if isinstance(a, chess.Promotion) else [] ),
            long_push = ( [a] if isinstance(a, chess.LongPush) else [] )
            )

    @classmethod
    def init_from_state_change(cls, sc:StateChange) -> Self :
        is_en_passant = isinstance(sc.action, chess.EnPassant)
        return cls(
            action = [sc],
            capture = ( [sc]
                        if (is_en_passant or bool(isinstance(sc.action, chess.Movement)
                                                  and sc.action.taken))
                        else [] ),
            en_passant = ( [sc] if is_en_passant else [] ),
            castle = ( [sc] if isinstance(sc.action, chess.Castle) else [] ),
            promotion = ( [sc] if isinstance(sc.action, chess.Promotion) else [] ),
            long_push = ( [sc] if isinstance(sc.action, chess.LongPush) else [] )
            )



# ---- exploration (algorithme perft)
class QuitExploration(Exception) :
    """Exception levée pour quitter l'exploration des possibilités en mode
    step by step
    """


def explore(game : chess.Chessgame,
            depth : int,
            data_type : type,
            done : chess.Action = None) -> ByTags :
    """Explore toutes les actions réalisables sur l'échiquier game jusqu'à un
    niveau de profondeur depth (selon l'algorithme perft). Renvoie uniquement
    les actions possibles à effectuer au niveau depth de profondeur.
    """
    if depth == 0 :
        return data_type.init_from_action(done)

    data = data_type()
    for act in game.possibilities() :
        save_representation(act)
        if act.do() :
            data.update( explore(game, depth-1, data_type, act) )
            act.undo()

    return data


def perft(game : chess.Chessgame, depth : int) :
    for i in range(depth) :
        t0 = perf_counter()
        result = explore(game, i, CountActions)
        t1 = perf_counter()
        print(f"depth {i} : {result} ; exectuted in {t1-t0} seconds")


def _construct_step_by_step_exploration(game:chess.Chessgame,
                                        depth:int,
                                        applied:StateChange) -> ListActions :
    if depth == 0 :
        return ListActions.init_from_state_change(applied)

    data = ListActions()
    for act in game.possibilities() :
        save_representation(act)
        state_change = StateChange(act)

        if state_change.apply() :
            data.update( _construct_step_by_step_exploration(game,
                                                             depth-1,
                                                             state_change) )
            state_change.unapply()

    return data


def explore_step_by_step(game:chess.Chessgame, depth:int, tag:str='action') :
    exploration = _construct_step_by_step_exploration(game, depth, None)

    for i, state_change in enumerate(getattr(exploration, tag), start=1) :
        print(f'\f\n{tag} : {i}')
        state_change.display()
        if input() in ('q', 'quit', 'exit') :
            raise QuitExploration



# ---- construction de l'arbre des possibilités
class Leaf :
    """
    action : chess.Action
        Action qui lie cette feuille avec son noeud parent.
    state : str
        Fenstring qui représente l'état de l'échiquier.
    """
    __slots__ = ('action', 'state', )

    def __init__(self, game:chess.Chessgame, action:chess.Action) :
        self.action : chess.Action = action
        self.state : str = fenstrings.create(game)

    def __repr__(self) :
        return f"<Leaf>\n\n{state_repr_from_action(self.state, self.action)}"

    def branches(self, leaf_condition, path) -> Iterator :
        if leaf_condition(self) :
            yield path + [self]


class Node :
    """
    action : chess.Action
        Action qui lie ce noeud avec son noeud parent.
    state : str
        Fenstring qui représente l'état de l'échiquier.
    linked_nodes : Union[List[Node], List[Leaf]]
        Liste des noeuds suivants auxquels ce noeud est lié
    """
    __slots__ = ('action', 'state', 'linked_nodes')

    def __init__(self, game:chess.Chessgame, action:chess.Action) :
        self.action : chess.Action = action
        self.state : str = fenstrings.create(game)
        self.linked_nodes : Union[List[Node], List[Leaf]] = []

    def __repr__(self) :
        return (f"<Node linked to {len(self.linked_nodes)} other node(s)>\n\n"
                f"{state_repr_from_action(self.state, self.action)}")

    def depth(self) -> int :
        """Renvoie la profondeur de l'arbre"""
        depth = 0
        node = self
        while isinstance(node, Node) :
            node = node.linked_nodes[0]
            depth += 1
        return depth

    def branches(self, leaf_condition, path) -> Iterator :
        """Renvoie les branches dont les feuilles (c.a.d les noeuds terminaux)
        vérifient la condition leaf_condition
        """
        path = path + [self]
        for node in self.linked_nodes :
            yield from node.branches(leaf_condition, path)


def _build_tree(game : chess.Chessgame,
                depth : int,
                done : chess.Action) -> Union[Node, Leaf] :
    if depth == 0 :
        return Leaf(game, done)

    node = Node(game, done)
    for act in game.possibilities() :
        save_representation(act)
        if act.do() :
            node.linked_nodes.append( _build_tree(game, depth-1, act) )
            act.undo()

    return node


def explore_branches_step_by_step(game : chess.Chessgame,
                                  depth : int,
                                  leaf_condition : Callable=None,
                                  display_state_variables : bool = False) :
    if leaf_condition is None :
        leaf_condition = lambda leaf: True

    tree = _build_tree(game, depth, None)
    for branch in tree.branches(leaf_condition, []) :
        print('\f')
        print(branch_repr(branch, state_variables=display_state_variables))
        if input() in ('q', 'quit', 'exit') :
            raise QuitExploration




class BugTrackerFenstrings :
    # niveaux 1, 2, 3, 4, 5 corrects
    position1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    # niveaux 1, 2, 3 corrects
    # FIXME : 56 roques de trop au niveau 4
    position2 = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -"


    # sûrement mal documentée : non-interprétable
    position3 = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -"


    # niveaux 1, 2, 3, 4, 5 corrects
    position4 = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"


    position5 = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"


if __name__ == '__main__' :
    FENSTRING = BugTrackerFenstrings.position2

    game = chess.Chessgame()
    fenstrings.load(game, FENSTRING)






