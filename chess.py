# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 17:25:36 2022

@author: Victor Laügt
"""

import abc
from dataclasses import dataclass
from typing import TypeVar, Union, List, Tuple, Iterator, Dict

# from MyTools import debug
# dbg = debug.DebugSpace(name='chess', trackers=True, print_msg=True, msg_history=True)

# couleurs des pièces
Colour = TypeVar('Colour')
BLACK: Colour = False
WHITE: Colour = True

# gauche et droite de l'échiquier (pour différencier la tour du grand roque de
# celle du petit roque)
Side = TypeVar('Side')
LONG: Side = False  # grand roque (à gauche)
SHORT: Side = True  # petit roque (à droite)

# états de victoire de l'échiquier
WinState = TypeVar('WinState')
NOT_FINISHED: WinState = 0
MAT: WinState = 1
NULL_PAT: WinState = 2
NULL_REPETITIONS: WinState = 3

# vecteurs unitaires de déplacement des pièces
N = -10 # vecteur unitaire pointant vers le nord
S = +10 # vecteur unitaire pointant vers le sud
W = -1  # vecteur unitaire pointant vers l'ouest
E = +1  # vecteur unitaire pointant vers l'est



# def is_valid_square(square) -> bool:
#     line, column = divmod(square, 10)
#     return 0 <= line <= 7 and 0 <= column <= 7


# ---- Actions des pièces
def castle(board, king, rook, king_vector, rook_vector):
    board[king.square] = EMPTY
    board[rook.square] = EMPTY
    king.square += king_vector
    rook.square += rook_vector
    board[king.square] = king
    board[rook.square] = rook


def longpush(board, pawn, vector):
    board[pawn.square] = EMPTY
    pawn.square += vector
    board[pawn.square] = pawn


class Action(abc.ABC):
    """Classe mère de toute les actions des pièces de l'échiquier"""
    __slots__ = ('_debug_repr', 'board', 'piece')

    def __repr__(self):
        if hasattr(self, '_debug_repr'):
            return self._debug_repr
        else:
            return super().__repr__()

    # GUI: pour savoir quelles cases doivent apparaître en surbrillance dans
    # la gui
    @abc.abstractmethod
    def target_square(self) -> int:
        """Renvoie la case servant d'input pour executer cette action dans une
        interface graphique
        """

    # ---- - vérificateur d'actions
    # GUI: pour transmettre uniquement des actions légales à la gui
    def is_check_safe(self) -> bool:
        """Renvoie True ssi l'action ne met ni ne laisse le roi de self.piece
        en échec
        """
        check_safety = not self._execute()
        self._clean()
        return check_safety


    # ---- - executeurs d'action
    # ENGINE: lorsqu'il essaie une action, le moteur d'échecs continue de
    # l'explorer à condition que sa méthode do() renvoie True
    # GUI: l'interface graphique effectue les inputs du joueur avec la méthode
    # do() mais n'utilise pas sa valeur de retour
    def do(self) -> bool:
        """Si cette action ne met ni ne laisse le roi de self.piece en échec,
        alors effectue l'action et renvoie True
        Sinon, n'effectue pas l'action et renvoie False
        """
        if self._execute():
            # l'action met ou laisse le roi en échec
            self._clean()
            return False
        else:
            # l'action ne met ni ne laisse le roi en échec
            self._finalize()
            return True

    @abc.abstractmethod
    def _execute(self):
        """Commence l'execution de l'action en modifiant la variable array de
        l'échiquier
        Si l'action met ou laisse le roi de self.piece en échec, alors renvoie
        une pièce qui met le roi en échec
        Sinon, renvoie None
        """

    @abc.abstractmethod
    def _clean(self):
        """Interrompt l'execution de l'action en rétablissant la valeur de
        la variable array de l'échiquier
        """

    @abc.abstractmethod
    def _finalize(self):
        """Termine l'execution de l'action en modifiant les autres variables
        de l'échiquier
        """
        board = self.board
        board.player = not board.player
        board.history.append(self)

    # ENGINE: lorsque le moteur d'échec change de branche dans l'arbre des
    # possibilités, il utilise la méthode undo() pour remonter l'arbre
    # GUI/SERVER: lorsqu'une demande d'annulation d'action est acceptée, la
    # méthode undo() est utilisée
    @abc.abstractmethod
    def undo(self) -> None:
        """Annule l'action après que son execution ai été terminée"""
        self._clean()
        board = self.board
        board.player = not board.player
        board.history.pop()


class Movement(Action):
    """Déplacement classique d'une pièce quelconque d'une case vers une autre,
    pouvant éventuellement capturer une pièce adverse
    """
    __slots__ = ('start', 'end', 'taken', 'en_passant_save')

    def __init__(self, board, piece, start, end):
        self.board = board
        self.piece = piece
        self.start = start
        self.end = end

    def target_square(self):
        return self.end

    def _execute(self):
        board, piece, end = self.board, self.piece, self.end
        self.taken = board[end]
        board[self.start] = EMPTY
        board[end] = piece
        piece.square = end
        return board.check(piece.colour)

    def _clean(self):
        board, piece, start = self.board, self.piece, self.start
        board[start] = piece
        board[self.end] = self.taken
        piece.square = start

    def _finalize(self):
        super()._finalize()
        board, taken = self.board, self.taken
        if taken:
            board.won_material[self.piece.colour].append(taken)
            taken.alive = False
        self.en_passant_save = board.en_passant
        board.en_passant = None

    def undo(self):
        super().undo()
        board = self.board
        if self.taken:
            board.won_material[self.piece.colour].pop().alive = True
        board.en_passant = self.en_passant_save


class LeftRookFirstMovement(Movement):
    """Premier déplacement d'une tour gauche, après lequel il lui devient
    impossible d'effectuer le grand roque
    """
    __slots__ = ('rook_castling_save',)

    def _finalize(self):
        super()._finalize()
        rook = self.piece
        rook_castling = self.board.rook_castling[rook.colour]
        self.rook_castling_save = rook_castling[LONG]
        rook_castling[LONG] = False
        rook.movement_type = Movement

    def undo(self):
        super().undo()
        rook = self.piece
        self.board.rook_castling[rook.colour][LONG] = self.rook_castling_save
        rook.movement_type = LeftRookFirstMovement


class RightRookFirstMovement(Movement):
    """Premier déplacement d'une tour droite, après lequel il lui devient
    impossible d'effectuer le petit roque
    """
    __slots__ = ('rook_castling_save',)

    def _finalize(self):
        super()._finalize()
        rook = self.piece
        rook_castling = self.board.rook_castling[rook.colour]
        self.rook_castling_save = rook_castling[SHORT]
        rook_castling[SHORT] = False
        rook.movement_type = Movement

    def undo(self):
        super().undo()
        rook = self.piece
        self.board.rook_castling[rook.colour][SHORT] = self.rook_castling_save
        rook.movement_type = RightRookFirstMovement


class KingFirstMovement(Movement):
    """Premier déplacement d'un roi, après lequel il lui devient impossible
    d'effectuer le grand ou le petit roque
    """
    __slots__ = ('rook_castling_save',)

    def _finalize(self):
        super()._finalize()
        rook_castling, colour = self.board.rook_castling, self.piece.colour
        self.rook_castling_save = rook_castling[colour].copy()
        rook_castling[colour] = [False, False]
        self.piece.movement_type = Movement

    def undo(self):
        super().undo()
        self.board.rook_castling[self.piece.colour] = self.rook_castling_save
        self.piece.movement_type = KingFirstMovement


class Promotion(Movement):
    """Déplacement classique d'un pion entrainant sa promotion"""
    __slots__ = ('into',)

    def __init__(self, board, piece, start, end, into:type):
        self.board = board
        self.piece = piece  # pion
        self.start = start  # case de départ (du pion)
        self.end = end      # case d'arrivée (de la pièce promue)
        self.into = into    # type de la pièce promue

    # GUI: permet à la gui de décider le type de la pièce promue
    def promote_into(self, ptype:type):
        if not issubclass(ptype, Piece):
            raise TypeError(
                "promote_into(ptype): ptype doit être un type de pièce"
                )
        if ptype is Pawn:
            raise TypeError("Impossible de promouvoir un pion en un pion")
        self.into = ptype

    def _execute(self):
        board = self.board
        self.taken = board[self.end]
        board[self.start] = EMPTY
        return board.check(self.piece.colour)

    def _clean(self):
        self.board[self.start] = self.piece

    def _finalize(self):
        super()._finalize()
        board, pawn, end = self.board, self.piece, self.end
        pawn.alive = False
        promoted = self.into(end, pawn.colour)
        board[end] = promoted
        board.pieces[pawn.colour].append(promoted)

    def undo(self):
        super().undo()
        self.board.pieces[self.piece.colour].pop()
        self.board[self.end] = self.taken
        self.piece.alive = True


class LongPush(Action):
    """Déplacement d'un pion de deux cases en ligne droite vers le côté
    adverse
    """
    __slots__ = ('en_passant_save')

    def __init__(self, board, pawn):
        self.board = board
        self.piece = pawn

    def target_square(self):
        pawn = self.piece
        return pawn.square + pawn.longpush_vector

    def _execute(self):
        board, pawn = self.board, self.piece
        longpush(board, pawn, pawn.longpush_vector)
        return board.check(pawn.colour)

    def _clean(self):
        board, pawn = self.board, self.piece
        longpush(board, pawn, -pawn.longpush_vector)

    def _finalize(self):
        super()._finalize()
        board, pawn = self.board, self.piece
        self.en_passant_save = board.en_passant
        board.en_passant = pawn.square - pawn.push_vector

    def undo(self):
        super().undo()
        self.board.en_passant = self.en_passant_save


class EnPassant(Action):
    """Prise en passant d'un pion"""
    __slots__ = ('start', 'end', 'taken',)

    def __init__(self, board, piece, start, end):
        self.board = board
        self.piece = piece
        self.start = start
        self.end = end

    def target_square(self):
        return self.end

    def _execute(self):
        board, pawn, end = self.board, self.piece, self.end
        board[self.start] = EMPTY
        board[end] = pawn
        pawn.square = end
        captured_square = end - pawn.push_vector
        self.taken = board[captured_square]
        board[captured_square] = EMPTY
        return board.check(pawn.colour)

    def _clean(self):
        board, pawn, start = self.board, self.piece, self.start
        board[self.end] = EMPTY
        board[start] = pawn
        pawn.square = start
        board[self.taken.square] = self.taken

    def _finalize(self):
        super()._finalize()
        board, taken_pawn = self.board, self.taken
        board.won_material[self.piece.colour].append(taken_pawn)
        taken_pawn.alive = False
        board.en_passant = None

    def undo(self):
        super().undo()
        board, pawn = self.board, self.piece
        taken_pawn = board.won_material[pawn.colour].pop()
        taken_pawn.alive = True
        board.en_passant = taken_pawn.square - taken_pawn.push_vector


class Castle(Action):
    """Roque d'un roi"""
    __slots__ = ('rook', 'king_vector', 'rook_vector', 'rook_castling_save',
                 'en_passant_save')

    def __init__(self, board, king, rook, king_vector, rook_vector):
        self.board = board
        self.piece = king
        self.rook = rook
        self.king_vector = king_vector
        self.rook_vector = rook_vector

    def target_square(self):
        return self.piece.square + self.king_vector

    def is_check_safe(self):
        return True

    def _execute(self):
        castle(self.board, self.piece, self.rook, self.king_vector, self.rook_vector)

    def _clean(self):
        castle(self.board, self.piece, self.rook, -self.king_vector, -self.rook_vector)

    def _finalize(self):
        super()._finalize()
        board, colour = self.board, self.piece.colour
        self.rook_castling_save = board.rook_castling[colour].copy()
        self.en_passant_save = board.en_passant
        board.rook_castling[colour] = [False, False]
        board.en_passant = None

    def undo(self):
        super().undo()
        board = self.board
        board.rook_castling[self.piece.colour] = self.rook_castling_save
        board.en_passant = self.en_passant_save



# ---- Objets de l'échquier
def sqr_range_is_empty(board, a, b) -> bool:
    """Renvoie True ssi toutes les cases entre a et b (excluses) sont vides"""
    return not any(board[square] for square in range(a+1, b))


def sqr_range_is_safe(board, a, b, colour) -> bool:
    """Renvoie True ssi toutes les cases entre a et b (incluses) ne sont pas
    menacées par l'adversaire de colour
    """
    return not any(board.threat(square, colour) for square in range(a, b+1))


def every_promotions(board, pawn, start, end):
    """Itère sur toutes les promotions du pion pawn partant de la case start
    vers la case end
    """
    return (Promotion(board, pawn, start, end, ptype) for ptype in PROMOTIONS_TYPES)


class ChessboardItem(abc.ABC):
    """Classe mère de tous les objets de l'échiquier, soit:
        - les cases vides (aucune pièce n'est posée dessus)

        - les cases exterieures (elles ne sont accesibles par aucune pièces car
        elles sont à l'exterieure de l'échiquier)

        - les pièces de l'échiquier
    """
    def __eq__(self, other):
        return isinstance(other, type(self))

    @abc.abstractmethod
    def can_be_covered_by(self, piece) -> bool:
        """Renvoie True ssi self peut être recouvert par la pièce piece lorsque
        elle se déplace.

        Propriétés: Pour A et B des instances de la classe Piece
            OUT.can_be_covered_by(A) == False
            EMPTY.can_be_covered_by(A) == True
            A.can_be_covered_by(B) == B.can_be_covered_by(A)
        """

    @abc.abstractmethod
    def is_enemy_of(self, piece) -> bool:
        """Renvoie True ssi piece est une pièce de la couleur de l'adversaire
        de self.

        Propriétés: Pour A et B des instances de la calsse Piece
            OUT.is_enemy_of(A) == False
            EMPTY.is_enemy_of(A) == False
            A.is_enemy_of(B) == B.is_enemy_of(A)
        """

class Empty(ChessboardItem):
    """Implémente les cases vides c.a.d les cases pouvant être recouvertes par
    toutes les pièces et n'étant les ennemies d'aucun joueur
    """
    def __repr__(self):
        return '..'

    @staticmethod
    def unicode_repr():
        return '.'

    def __bool__(self):
        return False

    def can_be_covered_by(self, piece):
        return True # Une case vide n'empèche jamais une pièce de se poser
                    # dessus

    def is_enemy_of(self, piece):
        return False # Une case vide est neutre donc n'est enemi ni des blancs
                     # ni des noirs


class Out(ChessboardItem):
    """Implémente l'exterieur de l'échiquier c.a.d les cases ne pouvant être
    recouvertes par aucune pièces et n'étant les ennemies d'aucun joueur
    """
    def __repr__(self):
        return '__'

    def __bool__(self):
        return False

    def can_be_covered_by(self, piece):
        return False # Une pièce ne peut pas sortir de l'échiquier

    def is_enemy_of(self, piece):
        return False # Une pièce enemie ne peut pas être en dehors de
                     # l'échiquier


@dataclass
class FictivePiece:
    square:int
    colour:Colour
    vectors:Tuple[int]


class Piece(ChessboardItem):
    """Classe mère de toute les pièces de l'échiquier

    Vocabulaire:
        On dit qu'une case S est accessible par une pièce P si
        la pièce P, en partant de sa position actuelle, peut se rendre sur la
        case S en effectuant un déplacement classique (c.a.d une action
        instance de Movement), sans se soucier de savoir si cela va mettre son
        roi en échec
    """
    symbol:str
    score:int

    def __init__(self, square, colour):
        self.square = square  # case occupée par la pièce
        self.colour = colour  # couleur de la pièce
        self.alive = True     # True => pièce en jeu ; False => pièce capturée

    def __repr__(self):
        if hasattr(self, '_debug_repr'):
            return self._debug_repr
        if self.colour:
            return 'w'+self.symbol
        else:
            return 'b'+self.symbol

    def __eq__(self, other):
        return super().__eq__(other) and self.colour == other.colour

    def can_be_covered_by(self, piece):
        return self.colour != piece.colour

    def is_enemy_of(self, piece):
        return self.colour != piece.colour

    # ENGINE: permet au moteur d'échecs de créer l'arbre des possibilités
    def actions(self, board) -> Iterator[Action]:
        """Itère sur toutes les actions de la pièce self sur l'échiquier board"""
        start = self.square
        return (Movement(board, self, start, end)
                for end in self._ending_squares(board))

    # GUI/SERVER: permet d'obtenir toutes les actions faisables par une pièce, dans un
    # format approprié pour la gui ou le serveur
    def choices(self, board) -> Dict[int, Action]:
        """self.choices(board)[end] == action qui, dans une interface graphique, s'effectue
        en déplaçant la pièce self vers la case end
        """
        return {act.target_square(): act
                for act in self.actions(board) if act.is_check_safe()}


class CastlingPiece:
    """Classe dont héritent les pièces impliquées dans le roque c.a.d Rook et
    King
    """
    def actions(self, board):
        start = self.square
        return (self.movement_type(board, self, start, end)
                for end in self._ending_squares(board))


# ---- - pions
class Pawn(Piece):
    """Implémente les pions
    Actions:
        Movement
        LongPush: déplacement de deux cases vers le côté adverse
        EnPassant: prise en passant
    """
    symbol = 'p'
    score = 1
    attacks = ((S+W, S+E), (N+W, N+E))

    def unicode_repr(self):
        return ('♙' if self.colour else '♟')

    def __init__(self, square, colour):
        super().__init__(square, colour)
        if colour:
            self.push_vector = N
            self.longpush_vector = 2*N
            self.allied_pawns_line = 60
            self.enemy_pawns_line = 10
        else:
            self.push_vector = S
            self.longpush_vector = 2*S
            self.allied_pawns_line = 10
            self.enemy_pawns_line = 60
        self.diag_vectors = self.attacks[colour]

    def is_in(self, line:int) -> bool:
        """Renvoie True ssi self est sur la ligne line"""
        return line <= self.square <= (line+7)

    def actions(self, board):
        square = self.square

        # Promotion
        if self.is_in(self.enemy_pawns_line):
            end = square + self.push_vector
            if board[end] is EMPTY:
                yield from every_promotions(board, self, square, end)
            for vector in self.diag_vectors:
                end = square + vector
                if board[end].is_enemy_of(self):
                    yield from every_promotions(board, self, square, end)

        else:
            # LongPush
            if (self.is_in(self.allied_pawns_line)
                    and board[square + self.push_vector] is EMPTY
                    and board[square + self.longpush_vector] is EMPTY):
                yield LongPush(board, self)

            # Movement | EnPassant
            end = square + self.push_vector
            if board[end] is EMPTY:
                yield Movement(board, self, square, end)
            for vector in self.diag_vectors:
                end = square + vector
                if board.en_passant == end:
                    yield EnPassant(board, self, square, end)
                elif board[end].is_enemy_of(self):
                    yield Movement(board, self, square, end)


# ---- - pièces glissantes
class SlidingPiece(Piece):
    """Implémente les pièces qui, dans un sens et une direction donnée, se
    déplacent d'une distance pouvant varier: Bishop, Rook, Queen
    """
    vectors:Tuple[int]

    def _ending_squares(self, board) -> Iterator[int]:
        """Itère sur les cases accessibles par la pièce self sur l'échiquier
        board
        """
        start = self.square
        for v in self.vectors:
            end = start + v
            piece = board[end]
            while piece.can_be_covered_by(self):
                yield end
                if piece:
                    break
                end += v
                piece = board[end]


class Bishop(SlidingPiece):
    """Implémente les fous
    Actions:
        Movement: déplacement classique en diagonale
    """
    symbol = 'b'
    score = 3
    vectors = (N+W, N+E, S+W, S+E)

    def unicode_repr(self):
        return ('♗' if self.colour else '♝')


class Rook(CastlingPiece, SlidingPiece):
    """Implémente les tours
    Actions:
        Movement: déplacement classique en ligne droite
    """
    symbol = 'r'
    score = 5
    vectors = (N, E, S, W)

    def unicode_repr(self):
        return ('♖' if self.colour else '♜')

    def __init__(self, square, colour, movement_type=Movement):
        super().__init__(square, colour)
        self.movement_type = movement_type


class Queen(SlidingPiece):
    """Implémente les dames
    Actions:
        Movement: déplacement classique en ligne droite et en diagonale
    """
    symbol = 'q'
    score = 9
    vectors = Rook.vectors + Bishop.vectors

    def unicode_repr(self):
        return ('♕' if self.colour else '♛')



# ---- - pièces sautantes
class JumpingPiece(Piece):
    """Implémente les pièces qui, dans un sens et une direction donnée, se
    déplacent toujours de la même distance: Knight, King
    """
    vectors:Tuple[int]

    def _ending_squares(self, board) -> Iterator[int]:
        """Itère sur les cases accessibles par la pièce self sur l'échiquier
        board
        """
        start = self.square
        for v in self.vectors:
            end = start + v
            if board[end].can_be_covered_by(self):
                yield end


class Knight(JumpingPiece):
    """Implémente les cavaliers
    Actions:
        Movement: déplacement classique en L
    """
    symbol = 'n'
    score = 3
    vectors = (2*N+W, 2*N+E, N+2*W, N+2*E, 2*S+W, 2*S+E, S+2*W, S+2*E)

    def unicode_repr(self):
        return ('♘' if self.colour else '♞')


class King(CastlingPiece, JumpingPiece):
    """Impélmente les rois
    Actions:
        Movement: déplacement classique de une case dans n'importe quelle direction
        Castle: grand ou petit roque
    """
    symbol = 'k'
    score = None
    vectors = Queen.vectors
    castling_vectors = ((2*W, 3*E), (2*E, 2*W))

    def unicode_repr(self):
        return ('♔' if self.colour else '♚')

    def __init__(self, square, colour):
        super().__init__(square, colour)
        if colour:
            self.rook_begining_squares = (70, 77)
        else:
            self.rook_begining_squares = (0, 7)
        self.movement_type = KingFirstMovement

    def actions(self, board):
        # Movement | KingFirstMovement
        yield from super().actions(board)

        # Castle
        colour, king_sqr = self.colour, self.square

        rook_castling = board.rook_castling[colour]
        l_rook, r_rook = board.rooks[colour]

        ((king_l_vct, l_rook_vct),
         (king_r_vct, r_rook_vct)) = self.castling_vectors

        if (
            rook_castling[LONG]
            and sqr_range_is_empty(board, l_rook.square, king_sqr)
            and sqr_range_is_safe(board, king_sqr+king_l_vct, king_sqr, colour)
            ):
                yield Castle(board, self, l_rook, king_l_vct, l_rook_vct)


        if (rook_castling[SHORT]
            and sqr_range_is_empty(board, king_sqr, r_rook.square)
            and sqr_range_is_safe(board, king_sqr, king_sqr+king_r_vct, colour)
            ):
                yield Castle(board, self, r_rook, king_r_vct, r_rook_vct)



# ---- Echiquier
EMPTY = Empty()
OUT = Out()
PROMOTIONS_TYPES = (Queen, Knight, Rook, Bishop)
NEIGHBORHOOD_EXPLORATION = (
    (JumpingPiece._ending_squares, Knight.vectors, Knight),
    (JumpingPiece._ending_squares, King.vectors, King),
    (SlidingPiece._ending_squares, Bishop.vectors, (Bishop, Queen)),
    (SlidingPiece._ending_squares, Rook.vectors, (Rook, Queen))
    )


def _first_line(colour) -> List[Piece]:
    """Crée et renvoie une liste contenant les pièces de la première ligne du
    joueur colour lors de l'initialisation de la partie
    """
    if colour:
        i = 70
    else:
        i = 0
    return [
        Rook(i+0, colour, LeftRookFirstMovement), Knight(i+1, colour),
        Bishop(i+2, colour), Queen(i+3, colour), King(i+4, colour),
        Bishop(i+5, colour), Knight(i+6, colour),
        Rook(i+7, colour, RightRookFirstMovement)
        ]

def _pawn_line(colour) -> List[Piece]:
    """Crée et renvoie une liste contenant 8 pions de la couleur colour"""
    if colour:
        i = 60
    else:
        i = 10
    return [Pawn(s, colour) for s in range(i, i+8)]


class Chessgame:
    """Implémente les échiquiers

    Chaque échiquier possède ses attributs constants:

    Attributs constants
    ----------------------
    kings: Tuple[King]
        board.kings[colour] == roi de la couleur colour sur l'échiquier board

    rooks: Tuple[Tuple[Rook]]
        board.rooks[colour][side] == tour initiale de la couleur colour et du
        côté side sur l'échiquier board
    ----------------------

    Un échiquier est une machine d'états dont chaque état est caractérisé par
    les attributs variables suivants:

    Attributs variables
    ----------------------
    array: List[ChessboardItem]
        board.array[square] == pièce sur la case square de l'échiquier board

    pieces: Tuple[List[Piece]]
        board.pieces[colour] == liste de toutes les pièces de la couleur colour

    rook_castling: List[List[bool]]
        board.rook_castling[colour][side] == True ssi la tour de la couleur
        colour et du côté side sur l'échiquier board ne s'est encore jamais
        déplacée

    en_passant: Union[None, int]
        Si une prise en passant est possible, alors en_passant est la position
        d'arrivée du pion qui effectue la prise en passant
        Sinon, en_passant vaut None


    won_material: Tuple[List[Piece]]
        board.won_material[colour] == liste des pièces que le joueur colour a
        capturé à son adversaire

    history: List[Action]
        Liste des actions qui ont été jouée sur l'échiquier

    player: Colour
        Couleur du joueur qui est en train d'effectuer une action
        False == joueur noir
        True == joueur blanc
    ----------------------
    """
    def __init__(self):
        white_first_line = _first_line(WHITE)
        white_pawn_line = _pawn_line(WHITE)
        black_pawn_line = _pawn_line(BLACK)
        black_first_line = _first_line(BLACK)

        # variables
        self.array = (
              black_first_line + 2*[OUT]
            + black_pawn_line + 2*[OUT]
            + 4*(8*[EMPTY] + 2*[OUT])
            + white_pawn_line + 2*[OUT]
            + white_first_line + 2*[OUT]
            + 19*[OUT]
            )
        self.pieces = (
            black_first_line + black_pawn_line,
            white_first_line + white_pawn_line
            )

        self.rook_castling = [[True, True], [True, True]]
        self.en_passant = None

        self.won_material = ([], [])
        self.history = []
        self.player = WHITE

        # constantes
        self.kings = (self[4], self[74])
        self.rooks = ((self[0], self[7]), (self[70], self[77]))

    # ---- - représentation de l'échiquier dans la console
    def _array_repr(self): # for debug only
        array_repr = []
        for i in range(0, 81, 10):
            array_repr.append(', '.join(map(str, self.array[i:i+10])))
        array_repr.append(', '.join(map(str, self.array[90:])))
        array_repr = '\n'.join(array_repr)
        return f'[\n{array_repr}\n]'

    def _info_repr(self):
        info_repr = []
        for colour, colour_name in ((WHITE, 'white'), (BLACK, 'black')):
            material = [type(piece) for piece in self.won_material[colour]]
            material_repr = set()
            for ptype in material:
                occurence = material.count(ptype)
                if occurence == 1:
                    material_repr.add(ptype.symbol)
                else:
                    material_repr.add(f'{occurence}*{ptype.symbol}')
            info_repr.append(
                f'    {colour_name} player\n'
                f'left castling: {self.rook_castling[colour][LONG]}\n'
                f'right castling: {self.rook_castling[colour][SHORT]}\n'
                f"won material: {', '.join(material_repr)}"
                )
        info_repr.append(
            '    next action\n'
            f"player: {'white' if self.player else 'black'}\n"
            f'en passant: {self.en_passant}'
            )
        return '\n\n'.join(info_repr)

    def _board_repr(self):
        b_repr = ['   A |B |C |D |E |F |G |H |']
        for j, num in zip(range(0, 80, 10), range(8, 0, -1)):
            line = self.array[j:j+8]
            b_repr.append(f"{num} |{'|'.join(map(str, line))}| ({j:02d}-{j+7:02d})")
        return '\n'.join(b_repr)

    def _board_unicode_repr(self):
        b_repr = ['  A|B|C|D|E|F|G|H|']
        unicode_repr = (lambda item: item.unicode_repr())
        for j, num in zip(range(0, 80, 10), range(8, 0, -1)):
            line = self.array[j:j+8]
            b_repr.append(f"{num}|{'|'.join(map(unicode_repr, line))}|({j:02d}-{j+7:02d})")
        return '\n'.join(b_repr)

    def __repr__(self):
        return (
            f"actions played: {len(self.history)}\n{self._board_repr()}\n\n"
            f"{self._info_repr()}"
            )

    def unicode_repr(self):
        return (
            f"actions played: {len(self.history)}\n{self._board_unicode_repr()}\n\n"
            f"{self._info_repr()}"
            )

    # ---- - comparaison d'états
    def eq_positions(self, other):
        return self.array == other.array

    def __eq__(self, other):
        return (isinstance(other, Chessgame)
                and self.eq_positions(other)
                and self.rook_castling == other.rook_castling
                and self.en_passant == other.en_passant
                and self.won_material == other.won_material
                and self.player == other.player)


    # ---- - lecture/écriture de l'échiquier
    def __getitem__(self, square) -> ChessboardItem:
        """board[square] == la pièce posée sur la case square de l'échiquier
        board
        """
        return self.array[square]

    def __setitem__(self, square, piece):
        self.array[square] = piece

    # GUI: l'interface graphique annule les actions grâce à la méthode
    # Chessgame.undo
    def undo(self) -> Union[Action, None]:
        """Annule la dernière action jouée si elle existe"""
        if self.history:
            action = self.history[-1]
            action.undo()
            return action

    # ---- - tests de menace
    def threat(self, square, colour) -> Union[Piece, None]:
        """Si la case square est menacée par une ou plusieurs pièces de
        l'adversaire du joueur colour, alors renvoie l'une de ces pièces.
        Sinon, renvoie None
        """
        # vérifie si square est menacée par un pion
        explorer = FictivePiece(square, colour, Pawn.attacks[colour])
        for s in JumpingPiece._ending_squares(explorer, self):
            item = self[s]
            if isinstance(item, Pawn):
                return item

        # vérifie si square est menacée par une grande pièce (c.a.d pas un pion)
        for square_finder, vectors, piece_type in NEIGHBORHOOD_EXPLORATION:
            explorer = FictivePiece(square, colour, vectors)
            for s in square_finder(explorer, self):
                item = self[s]
                if isinstance(item, piece_type):
                    return item

    def check(self, colour) -> Union[Piece, None]:
        """Si le roi du joueur colour est mis en échec par une ou plusieurs
        pièces de son adversaire, alors renvoie l'une de ces pièces.
        Sinon, renvoie None
        """
        return self.threat(self.kings[colour].square, colour)

    # ---- - scénarios
    def null_by_repetitions(self) -> bool: # TODO: Chessgame.null_by_repetitions()
        ...

    # GUI: la gui utilise la méthode choices pour obtenir la liste des actions
    # légales de chaque pièces
    def choices(self) -> Tuple[WinState, Dict[int, Dict[int, Action]]]:
        """Renvoie le tuple (win_state, choices)

        win_state: WinState
            NOT_FINISHED: partie non-terminée
            MAT: échec et mat
            NULL_PAT: nulle par pat
            NULL_REPETITIONS: nulle par répétitions

        choices: Dict[int, Dict[int, Action]]
            choices[start][end] == action qui, dans une interface graphique,
            s'effectue en déplaçant la pièce située sur la case start vers la
            case end
        """
        if self.null_by_repetitions():
            return (NULL_REPETITIONS, {})

        moving_piece_choices = {}
        for piece in self.pieces[self.player]:
            if piece.alive:
                action_choice = piece.choices(self)
                if action_choice:
                    moving_piece_choices[piece.square] = action_choice

        # Il existe au moins une action jouable
        if moving_piece_choices:
            return (NOT_FINISHED, moving_piece_choices)

        # Plus aucune action jouable
        else:
            if self.check(self.player):
                return (MAT, moving_piece_choices)
            else:
                return (NULL_PAT, moving_piece_choices)


    # ENGINE: le moteur d'échecs utilise la méthode possibilities() pour créer
    # l'arbre des possibilités: seules les actions dont la méthode do()
    # renvoie True sont légales
    def possibilities(self) -> List[Action]:
        """Renvoie la liste de toutes les actions du joueur self.player"""
        return [
            act for piece in self.pieces[self.player] if piece.alive
            for act in piece.actions(self)
            ]



# ---- Caractéristiques des actions
# GUI: la gui utilise ces fonctions pour obtenir certaines caractéristiques
# des actions qu'elle effectue ou annule. Ces informations permettent à la gui
# de créer les animations.
def player_colour(action:Action) -> Colour:
    return action.piece.colour

def enemy_colour(action:Action) -> Colour:
    return not action.piece.colour

def castle_side(castle:Castle) -> Side:
    return (castle.king_vector, castle.rook_vector) == King.castling_vectors[SHORT]

def start_square(action:Union[Movement, LongPush, EnPassant]) -> int:
    if isinstance(action, LongPush):
        return action.piece.square
    return action.start

def end_square(action:Union[Movement, LongPush, EnPassant]) -> int:
    return action.target_square()

def captured_square(action:Union[Movement, EnPassant]) -> int:
    return action.taken.square

def captured_piece(action:Union[Movement, EnPassant]) -> Union[Piece, Empty]:
    return action.taken




if __name__ == '__main__':
    game = Chessgame()
