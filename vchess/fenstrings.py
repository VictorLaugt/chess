# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 18:09:37 2022

@author: Victor Laügt

Implémente des fonctions permettant de
- créer la fenstring décrivant l'état courant d'un échiquier
- restaurer un état de l'échiquier à partir d'une fenstring
"""

from vchess import chess

from typing import Dict

__all__ = ['load', 'loaded', 'create']

# from MyTools import debug
# dbg = debug.DebugSpace(name='fenstrings', trackers=True, print_msg=True, msg_history=True)

EMPTY = chess.EMPTY
BLACK, WHITE = chess.BLACK, chess.WHITE
LONG, SHORT = chess.LONG, chess.SHORT

FEN_BLANK_SQR = '12345678'
PIECE_TYPES = {
    'p':chess.Pawn,
    'r':chess.Rook,
    'n':chess.Knight,
    'b':chess.Bishop,
    'q':chess.Queen,
    'k':chess.King
    }


# Uniquement pour le debugage
# def _not_consistent(game:chess.Chessgame):
#     for pieces in game.pieces:
#         for p in pieces:
#             if p.alive and p != game[p.square]:
#                 return p

# def _consistent_test(game:chess.Chessgame):
#     if _not_consistent(game):
#         comp_game = chess.Chessgame()
#         for i in range(0, 8):
#             for j in range(0, 8):
#                 comp_game[i*10 + j] = chess.EMPTY
#         for pieces in game.pieces:
#             for p in pieces:
#                 if p.alive:
#                     comp_game[p.square] = p
#         print('\n'.join(repr(comp_game).splitlines()[1:10]))
#     else:
#         print('game is consistent')

# def piece_eq(x, y):
#     if isinstance(x, chess.Empty) or isinstance(y, chess.Empty):
#         return type(x) == type(y)

#     elif isinstance(x, chess.Out) or isinstance(y, chess.Out):
#         return type(x) == type(y)

#     elif isinstance(x, chess.Piece) and isinstance(y, chess.Piece):
#         return (
#             type(x) == type(y)
#             and x.colour == y.colour
#             )

# def array_eq(a, b):
#     for sqr_id, (x, y) in enumerate(zip(a, b)):
#         if not piece_eq(x, y):
#             print(sqr_id, x, y)
#             return  False
#     return True

# def fen_is_valid(fen:str, game:chess.Chessgame) -> bool:
#     return array_eq(loaded(fen).array, game.array)

# def test(game:chess.Chessgame, depth:int):
#     if depth == 0:
#         return 1, None

#     count = 0
#     for act in game.possibilities():
#         if act.do():
#             add = test(game, depth-1)[0]
#             count += add

#             if not fen_is_valid(create(game), game):
#                 return count, create(game)

#             act.undo()
#     return count, None


# ---- squares
COLUMN_NAMES = 'abcdefgh'
COLUMN_INDEXES = {letter:i for i, letter in enumerate(COLUMN_NAMES)}

def name_to_square_id(name:str) -> int:
    """Renvoie l'identifaint de la case dont le nom est name"""
    letter, number = name
    return 10*(8-int(number)) + int(COLUMN_INDEXES[letter])


def indexes_to_square_id(i:int, j:int) -> int:
    """Renvoie l'identifiant de la case dont les indices de ligne et de colonne
    sont respectivement i et j
    """
    return 10*i + j


def square_id_to_name(sqr_id:int) -> str:
    """Renvoie le nom de la case dont l'identifiant est sqr_id"""
    return COLUMN_NAMES[sqr_id % 10] + str(8 - sqr_id//10)


# ---- read fenstrings
def new_piece(symbol:str, square:int) -> chess.Piece:
    """Crée et renvoie une pièce dont le symbole dans la convention fen est
    symbol et dont la position initiale est square
    """
    colour = symbol.isupper()
    stype = PIECE_TYPES[symbol.lower()]
    return stype(square, colour)


def get_pieces(game:chess.Chessgame) -> Dict[str, chess.Piece]:
    """Renvoie un dictionnaire dont les clés sont les symboles des pièces dans
    la convention fen et dont les valeurs sont les pièces correspondantes sur
    l'échiquier game
    """
    # inversion des listes avec un slice [x:y:-1] pour pouvoir ensuite les
    # vider à l'endroit avec .pop()
    black_first_line = game.pieces[BLACK][7::-1]
    black_pawn_line = game.pieces[BLACK][15:7:-1]
    white_first_line = game.pieces[WHITE][7::-1]
    white_pawn_line = game.pieces[WHITE][15:7:-1]

    #dbg.dprint(black_first_line, black_pawn_line, white_first_line, white_pawn_line,
    #      sep='\n', end='\n'*2)

    pieces = {'p':black_pawn_line, 'P':white_pawn_line}
    for font, first_line in ((str.lower, black_first_line),
                             (str.upper, white_first_line)):
        pieces[font('r')] = [first_line[0], first_line[7]]
        pieces[font('n')] = [first_line[1], first_line[6]]
        pieces[font('b')] = [first_line[2], first_line[5]]
        pieces[font('q')] = [first_line[4],] # attention: listes inversées
        pieces[font('k')] = [first_line[3],] # attention: listes inversées
    return pieces


def init_positions(game:chess.Chessgame, lines_field:str):
    pieces_dct = get_pieces(game)

    # met des valeurs par défaut à game.rook_castling et game.en_passant au cas
    # où les champs castles et en_passant ne sont pas renseignés dans la
    # fenstring
    black_left_rook, black_right_rook = pieces_dct['r']
    white_left_rook, white_right_rook = pieces_dct['R']

    game.rook_castling[BLACK][LONG] =\
        (black_left_rook.alive and black_left_rook.square == 0)

    game.rook_castling[BLACK][SHORT] =\
        (black_right_rook.alive and black_right_rook.square == 7)

    game.rook_castling[WHITE][LONG] =\
        (white_left_rook.alive and white_left_rook.square == 70)

    game.rook_castling[WHITE][SHORT] =\
        (white_right_rook.alive and white_right_rook.square == 77)

    game.en_passant = None

    # modifie l'attribut .square de chaque pièces
    for i, line in enumerate(lines_field.split('/')):
        j = 0
        for symbol in line:
            if symbol in FEN_BLANK_SQR:
                k = j + int(symbol)
                for sqr_id in range(indexes_to_square_id(i, j),
                                    indexes_to_square_id(i, k)):
                    game.array[sqr_id] = EMPTY
                j = k
            else:
                pieces_lst = pieces_dct[symbol]
                if pieces_lst:
                    p = pieces_lst.pop()
                    p.square = indexes_to_square_id(i, j)
                    p.alive = True
                else:
                    p = new_piece(symbol, indexes_to_square_id(i, j))
                    game.pieces[p.colour].append(p)
                game.array[p.square] = p
                j += 1
    for pieces_lst in pieces_dct.values():
        for p in pieces_lst:
            p.alive = False


def init_colour(game:chess.Chessgame, colour_field:str):
    game.player = (colour_field == 'w')


def init_castles(game:chess.Chessgame, castles_field:str):
    for symbol in castles_field:
        if symbol != '-':
            colour = symbol.isupper()
            side = (symbol.lower() == 'k')
            game.rook_castling[colour][side] = True


def init_en_passant(game:chess.Chessgame, en_passant_field:str):
    if en_passant_field == '-':
        game.en_passant = None
    else:
        game.en_passant = name_to_square_id(en_passant_field)


def init_actions_nb(game:chess.Chessgame, actions_nb_field:str):
    ... # TODO: init_actions_nb


def init_n(game:chess.Chessgame, n_field:str):
    ... # TODO: init_n


def load(game:chess.Chessgame, fenstring:str):
    """Met l'échiquier game dans l'état décrit par la fenstring reçue en
    paramètre.

    ATTENTION: TODO: améliorer cet aspect
        load() peut être vue comme une fonction qui trie les pièces de
        l'échiquier `game` selon une relation d'ordre qui est décrite par
        la fenstring `fenstring`.  Cependant, ce tri n'est pas stable, c'est
        à dire qu'il ne préserve pas l'ordonnancement initial des éléments que
        la relation d'ordre considère comme égaux.
        Donc, même si `fenstring` est une fenstring représentant l'état actuel
        de l'échiquier `game`, il est possible que ``load(game, fenstring)``
        permute certaines pièces de `game` qui sont de la même nature.
        Dans ce cas, ``load(game, fenstring); game.undo()`` risque d'entrainer
        des bugs.
    """
    fields = fenstring.split()
    fields_lenght = len(fields)
    if fields_lenght >= 1:
        init_positions(game, fields[0])

    if fields_lenght >= 2:
        init_colour(game, fields[1])

    if fields_lenght >= 3:
        init_castles(game, fields[2])

    if fields_lenght >= 4:
        init_en_passant(game, fields[3])

    if fields_lenght >= 5:
        init_actions_nb(game, fields[4])

    if fields_lenght >= 6:
        init_n(game, fields[5])


def loaded(fenstring:str) -> chess.Chessgame:
    """Crée et renvoie un échiquier dans l'état décrit par la fenstring reçue
    en paramètre.
    Identité: loaded(create(game)) == game
    """
    game = chess.Chessgame()
    load(game, fenstring)
    return game


# ---- write fenstrings
def get_positions(game:chess.Chessgame) -> str:
    positions = []
    for i in range(0, 80, 10):
        line = []
        empty_squares = 0
        for j in range(0, 8):
            item = game[i+j]
            if item == chess.EMPTY:
                empty_squares += 1
            else:
                if empty_squares != 0:
                    line.append(str(empty_squares))
                    empty_squares = 0
                line.append(item.symbol.upper() if item.colour else item.symbol.lower())
        if empty_squares != 0:
            line.append(str(empty_squares))
        positions.append(''.join(line))
    return '/'.join(positions)


def get_colour(game:chess.Chessgame) -> str:
    if game.player:
        return 'w'
    else:
        return 'b'


def get_castles(game:chess.Chessgame) -> str:
    letters = []
    if game.rook_castling[WHITE][SHORT]: letters.append('K')
    if game.rook_castling[WHITE][LONG]: letters.append('Q')
    if game.rook_castling[BLACK][SHORT]: letters.append('k')
    if game.rook_castling[BLACK][LONG]: letters.append('q')
    return ''.join(letters) if letters else '-'


def get_en_passant(game:chess.Chessgame) -> str:
    if game.en_passant is None:
        return '-'
    else:
        return square_id_to_name(game.en_passant)


def get_actions_nb(game:chess.Chessgame) -> str:
    ... # TODO: get_actions_nb
    return '0'


def get_n(game:chess.Chessgame) -> str:
    ... # TODO: get_n
    return '1'


def create(game:chess.Chessgame) -> str:
    """Renvoie la fenstring permettant de décrire l'état actuel de l'échiquier
    game.
    Identité: create(loaded(fenstring)) == fenstring
    """
    return ' '.join((
        get_positions(game),
        get_colour(game),
        get_castles(game),
        get_en_passant(game),
        get_actions_nb(game),
        get_n(game)
        ))


if __name__ == "__main__":
    f0 = "8/8/8/8/8/8/8/8 w - - 0 1"
    f1 = "rn2kb1r/2ppPp1p/p4np1/1p2p3/1PQ5/2N2P1N/P3P1PP/R1B1KB1R b KQkq - 0 1"
    f2 = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -"
    promotion_test_fenstring = "rnbqkb1r/pppppp1p/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    castle_test_fenstring = "rnbqkbnr/8/8/8/8/8/8/RNBQKBNR b KQkq - 0 1"

    game = chess.Chessgame()
    load(game, f2)
