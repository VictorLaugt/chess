# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 17:49:42 2022

@author: Victor Laügt

Interface graphique minimaliste, spécialement conçue pour debuguer.
"""
__all__ = ['start_debug_ui']


import chess
import fenstrings

import threading
import tkinter as tk
from PIL import Image, ImageTk

from enum import IntEnum
from typing import Union, Tuple, List, Dict, Iterable

# from MyTools import debug
# dbg = debug.DebugSpace(name='debug_ui', trackers=True, print_msg=True, msg_history=True)

ActionsDictionary = Dict[int, chess.Action]
ChoicesDictionary = Dict[int, ActionsDictionary]

# types de marques
class MarkType(IntEnum):
    PIECE_CHOICE = 0
    ACTION_CHOICE = 1
    ERROR = 2

# Typographie
FONT = 'DejaVu Sans Mono'
BOLD_FONT = (FONT, 10, "bold")
NORMAL_FONT = (FONT, 10)

# couleurs des joueurs
BLACK = False
WHITE = True

# types des pièces de l'échiquier
PIECE_TYPES = (chess.Pawn, chess.Bishop, chess.Knight, chess.Rook, chess.Queen,
               chess.King)

MAIN_BLUE = "#5D5D47"
WSQR_COLOR = "#FFFFD3"
BSQR_COLOR = "#8B8B73"


# noms des fichiers images des pièces
IMAGE_PATH = "./_images"
IMAGE_NAMES = {
    chess.Pawn: "/pion.png",
    chess.Bishop: "/fou.png",
    chess.Knight: "/cavalier.png",
    chess.Rook: "/tour.png",
    chess.Queen: "/dame.png",
    chess.King: "/roi.png",
    }


# ---- interface entre l'ui et le noyau
def square_id(i:int, j:int) -> int:
    """square_id(i, j) == identifiant dans le noyau de la case dont les indices
    de ligne et de colonne sont respectivement i et j
    """
    if 0 <= i <= 7 and 0 <= j <= 7:
        return i*10 + j
    else:
        raise IndexError

def square_indexes(sqr_id:int) -> Tuple[int]:
    """square_indexes(sqr_id) == couple d'indices (ligne i, colonne j) de la
    case dont l'identifiant est sqr_id dans le noyau
    """
    return divmod(sqr_id, 10)

def square_name(sqr_id:int) -> str:
    """square_name(sqr_id) == nom de la case dont l'identifiant est sqr_id dans
    le noyau
    """
    i, j = square_indexes(sqr_id)
    return 'abcdefgh'[j] + str(8-i)

def win_state_name(win_state:chess.WinState) -> Union[str, None]:
    """win_state_name(win_state) == chaine de caractères qui décrit l'état de
    la partie
    """
    for name in ('NOT_FINISHED', 'MAT', 'NULL_PAT', 'NULL_REPETITIONS'):
        if win_state == getattr(chess, name):
            return name


class CoreInterface:
    """Fait l'interface entre les inputs de l'utilisateur sur l'ui et la partie
    d'échec calculée par le noyau
    """
    def __init__(self, core:chess.Chessgame):
        self.core:chess.Chessgame = core
        self.selected_square:int = None
        self.win_state:chess.WinState = None
        self.choices:ChoicesDictionary = {}
        self.actions:ActionsDictionary = {}

        self.compute_choices()

    def search_all_errors(self) -> List[int]:
        """Cherche et renvoie les pièces erronnées:
        - Pièce mal positionnée sur l'échiquier
        - Pièce présente sur l'échiquier mais non vivante
        - Pièce présente sur l'échiquier mais non référencée
        - Pièce référencée et vivante mais non présente sur l'échiquier
        """
        core = self.core
        errors = []
        for i in range(0, 80, 10):
            for sqr_id in range(i, i+8):
                item = core[sqr_id]
                if isinstance(item, chess.Piece):
                    if (item.square != sqr_id
                            or not item.alive
                            or item not in core.pieces[item.colour]):
                        errors.append(item.square)
        for pieces in core.pieces:
            for p in pieces:
                if p.alive and p not in core.array:
                    errors.append(item.square)
        return errors

    def square_data(self, sqr_id:int) -> str:
        """Renvoie une description des attributs de la pièce située sur la case
        dont l'identifiant est sqr_id
        """
        item = self.core[sqr_id]
        if not isinstance(item, chess.Piece):
            return repr(item)
        data_repr = [
            f"{item} ({type(item).__name__})",
            f"square = {item.square} ({square_name(item.square)})",
            f"colour = {item.colour}",
            f"alive = {item.alive}",
            ]
        if item in self.core.pieces[item.colour]:
            data_repr.append("registered piece")
        else:
            data_repr.append("unregistered piece")
        return '\n'.join(data_repr)

    def compute_choices(self):
        self.win_state, self.choices = self.core.choices()

    def get_win_state(self) -> str:
        return win_state_name(self.win_state)

    def selected(self) -> bool:
        """Renvoie True ssi une case a été selectionnée par l'utilisateur"""
        return self.selected_square is not None

    def marked_squares(self) -> (Iterable[int], MarkType):
        """Renvoie le liste des cases à surligner et la couleur dans laquelle
        il faut les surligner
        """
        if self.selected():
            return self.actions.keys(), MarkType.ACTION_CHOICE
        else:
            return self.choices.keys(), MarkType.PIECE_CHOICE

    def select(self, sqr_id:int, input_repr:str):
        """Sélectionne la case dont l'identifiant est sqr_id"""
        actions = self.choices.get(sqr_id)
        if actions is not None:
            self.selected_square = sqr_id
            self.actions = actions
            return square_name(sqr_id)
        return input_repr+" (not available)"

    def unselect(self):
        """Désélectionne la case sélectionnée"""
        self.selected_square = None
        self.actions = {}

    def user_input(self, sqr_id:int) -> str:
        """Interprète les inputs de l'utilisateur lorsqu'il clique sur
        l'échiquier

        Prend en argument l'identifiant de la case sur laquelle l'utilisateur a
        cliqué. En fonction de la case actuellement cliquée et de la case
        précédement cliquée, fait l'une des choses ci-dessous:
            - sélectionner une pièce
            - désélectionner une pièce
            - déplacer la pièce selectionnée

        Renvoie une représentation de l'input
        """
        # sélection d'une pièce
        if self.selected_square is None:
            sqr_name = square_name(sqr_id)
            return self.select(sqr_id, sqr_name)

        # désélection de la pièce selectionnée
        elif self.selected_square == sqr_id:
            self.unselect()
            return ""

        # choix de l'action à effectuer avec la pièce selectionnée
        else:
            input_repr = f"{square_name(self.selected_square)}->{square_name(sqr_id)}"
            act = self.actions.get(sqr_id)
            if act is not None:
                act._debug_repr = action_repr(act)
                act.do()
                self.compute_choices()
                self.unselect()
                return input_repr
            else:
                return self.select(sqr_id, input_repr)

    def undo_action(self):
        """Annule la dernière action effectuée sur l'échiquier"""
        self.unselect()
        self.core.undo()
        self.compute_choices()


# ---- représentation des actions
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


def choices_repr_lines(coreinterface:CoreInterface) -> (List[str], List[int]):
    actions_list = [
        act for actions in coreinterface.choices.values()
        for act in actions.values()
        ]

    lines = [
        f'{i+1}: {action_repr(act)}' for i, act in enumerate(actions_list)
        ]
    if coreinterface.selected():
        bold_lines = [
            actions_list.index(act) for act in coreinterface.actions.values()
            ]
    else:
        bold_lines = []

    return lines, bold_lines


# ---- sprites
class Sprite(ImageTk.PhotoImage):
    def __init__(self, path:str, width:int):
        img = Image.open(path)
        l_img = img.size[0]
        if l_img != width:
            img = img.resize((width, width))
        super().__init__(img)


def piece_sprite(piece_type:type, colour:bool, width:int) -> Sprite:
    """Crée et renvoie un Sprite de largeur width représentant une pièce de
    type piece_type et de la couleur colour
    """
    pieces_dir = "/_effective_pieces"
    if colour:
        colour_dir = "/_white"
    else:
        colour_dir = "/_black"
    path = IMAGE_PATH + pieces_dir + colour_dir + IMAGE_NAMES[piece_type]
    return Sprite(path, width)


# ---- widgets
class TextBox(tk.Text):
    START = '1.0'
    END = 'end'

    def __init__(self, master, title:str=None, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.title = f'--- {title} ---\n' if title else ''

    def _display_title(self):
        self.insert('1.0', self.title)
        self.tag_add('title', '1.0', '2.0')
        self.tag_configure('title', font=BOLD_FONT)

    def display_text(self, txt:str, bold_start=None, bold_stop=None, /):
        """Efface le contenu de self puis affiche le texte txt. Les caractères
        dont les indices sont compris entre bold_start et bold_stop sont
        affichés en gras.
        """
        self.delete(self.START, self.END)
        self.insert(self.END, txt)
        self.tag_delete('highlight')
        if bold_stop:
            self.tag_add('highlight', bold_start, bold_stop)
            self.tag_configure('highlight', font=BOLD_FONT)
        self._display_title()

    def display_lines(self, lines:Iterable[str], bold_lines:Iterable[int]=()):
        """Efface le contenu de self puis affiche le texte dont les lignes sont
        stockées dans la liste lines. Les lignes dont les indices sont stockés
        dans le paramètre bold_lines sont affichées en gras.
        """
        self.delete(self.START, self.END)
        self.tag_delete('highlight')
        for i, l in enumerate(lines):
            self.insert(self.END, l+'\n')
            if i in bold_lines:
                self.tag_add('highlight', f'{i+1}.0', f'{i+2}.0')
        self.tag_configure('highlight', background='yellow', font=BOLD_FONT)
        self._display_title()

    def clear(self):
        """Efface le contenu de self"""
        self.delete(self.START, self.END)
        self._display_title()


class BoardViewer(tk.Canvas):
    """Affichage des cases et des pièces de l'échiquier"""
    def __init__(self, master):
        super().__init__(master)
        board_size = self.sqr_size * 9
        self.configure(width=board_size, height=board_size, bg=MAIN_BLUE)

        self.init_sprites()
        self.init_edges()

    @property
    def sqr_size(self) -> int:
        """Longueur du coté de chaque case de l'échiquier"""
        # height = self.master.winfo_height()
        # return round(height/10)
        return 70

    @property
    def coeff(self) -> float:
        return self.sqr_size / 80

    @property
    def marge(self) -> int:
        return round(30*self.coeff)

    def init_sprites(self):
        """Crée les sprites à afficher sur l'échiquier"""
        sprite_size = self.sqr_size
        self.mark_sprite = (
            Sprite(
                IMAGE_PATH + "/green_highlight.png",
                sprite_size
                ),
            Sprite(
                IMAGE_PATH + "/yellow_highlight.png",
                sprite_size
                ),
            Sprite(
                IMAGE_PATH + "/red_highlight.png",
                sprite_size
                )
            )
        self.piece_sprite = {}
        for ptype in PIECE_TYPES:
            self.piece_sprite[ptype] = (
                piece_sprite(ptype, BLACK, sprite_size), # sprite noir
                piece_sprite(ptype, WHITE, sprite_size)  # sprite blanc
                )

    def init_edges(self):
        """Affiche les bordures de l'échiquier"""
        font_size = str(round(18*self.coeff))
        marge = self.marge

        # Affichage des numéros des cases
        for i in range(8):
            y = (i+1) * self.sqr_size
            number = str(8-i)
            for (x, anch) in ((9*self.sqr_size-marge, tk.W), (marge, tk.E)):
                self.create_text(
                    x,
                    y,
                    text=number,
                    anchor=anch,
                    font=(FONT, font_size),
                    fill=WSQR_COLOR,
                    tags="edge_number"
                    )

        # Affichage des lettres des cases:
        for i, letter in zip(range(8), "ABCDEFGH"):
            x = (i+1) * self.sqr_size
            for (y, anch) in ((9*self.sqr_size-marge, tk.N), (marge, tk.S)):
                self.create_text(
                    x,
                    y,
                    text=letter,
                    anchor=anch,
                    font=(FONT, font_size),
                    fill=WSQR_COLOR,
                    tags="edge_letter"
                    )

    def get_piece_sprite(self, piece_type:type, colour:bool) -> Sprite:
        """Sprite de la pièce du type piece_type et de la couleur colour"""
        return self.piece_sprite[piece_type][colour]

    def coords_from_indexes(self, i:int, j:int) -> Tuple[int]:
        """board.coords_from_indexes(i, j) == coordonnées (x, y) du sprite
        représentant la pièce située à la case d'indices (i, j) sur l'échiquier
        board
        """
        marge = self.marge
        return (j*self.sqr_size + 1.5*marge, i*self.sqr_size + 1.5*marge)

    def indexes_from_coords(self, x:int, y:int) -> Tuple[int]:
        """board.indexes_from_coords(x, y) == indices (ligne i, colonne j) de la
        case dont le carré représentatif a pour coordonnée (x, y) dans le
        canvas
        """
        sqr_size = self.sqr_size
        marge = self.marge
        return int((y-1.5*marge) // sqr_size), int((x-1.5*marge) // sqr_size)

    def display_squares(self):
        """Actualise l'affichage des cases de l'échiquier"""
        # Efface les cases de l'échiquier et leur contenu
        self.delete("square")
        self.delete("piece")

        # Affiche les cases et leur contenu actuel
        for i in range(8):
            for j in range(8):
                # case (i, j)
                if (i+j) % 2 == 0:
                    sqr_colour = WSQR_COLOR
                else:
                    sqr_colour = BSQR_COLOR
                rectangle_points = (
                    *self.coords_from_indexes(i, j),
                    *self.coords_from_indexes(i+1, j+1)
                    )
                self.create_rectangle(
                    rectangle_points,
                    fill=sqr_colour,
                    width=0,
                    tags=("square", f"s{i}{j}", f"{sqr_colour}")
                    )

                # éventuelle pièce sur la case (i, j)
                x, y = self.coords_from_indexes(i, j)
                sqr_id = square_id(i, j)
                piece = self.master.core[sqr_id]
                if piece:
                    sprite = self.get_piece_sprite(type(piece), piece.colour)
                    self.create_image(
                        x,
                        y,
                        image=sprite,
                        anchor=tk.NW,
                        tags=("piece", f"p{i}{j}")
                        )
        self.tag_raise("piece")

    def display_marks(self, marks:Iterable[int], mark_type:MarkType):
        """Actualise l'affichage des marques sur les cases de l'échquier"""
        if mark_type is MarkType.PIECE_CHOICE:
            self.delete("mark")
        else:
            self.delete(mark_type.name)
        tag = mark_type.name

        for sqr_id in marks:
            i, j = square_indexes(sqr_id)
            x, y = self.coords_from_indexes(i, j)
            sprite = self.mark_sprite[mark_type]
            self.create_image(
                x,
                y,
                image=sprite,
                anchor=tk.NW,
                tags=("mark", tag, f"m{i}{j}")
                )
        self.tag_raise("mark")


class GameViewer(tk.Tk):
    """Fenêtre principale de l'interface graphique"""
    def __init__(self, core:chess.Chessgame, fenstr:str=None):
        super().__init__()
        self.core = core
        self.fenstr = fenstr
        if fenstr:
            fenstrings.load(self.core, fenstr)
        self.core_interface = CoreInterface(self.core)

        # apparence
        self.title("Chess Viewer")
        self.geometry(f"{905}x{770}+{0}+{0}")
        self.minsize(905, 770) # Taille minimale
        self.focus_force() # Forçage du focus sur la fenêtre
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # widgets à droite de l'échiquier
        right_frame = tk.Frame(self)
        self.core_state = TextBox(right_frame, 'core state', height=30, width=38)
        self.possibilities = TextBox(right_frame, 'possible actions', height=23, width=38)

        self.core_state.configure(font=NORMAL_FONT)
        self.possibilities.configure(font=NORMAL_FONT)

        self.core_state.pack()
        self.possibilities.pack()
        right_frame.pack(side=tk.RIGHT, anchor=tk.N+tk.W)

        # échiquier
        self.board = BoardViewer(self)
        self.board.bind('<Button-1>', self.click)            # click gauche
        self.board.bind('<Button-3>', self.show_square_data) # click droit
        self.board.pack(side=tk.TOP)

        # widgets en dessous de l'échiquier
        bottom_left_frame = tk.Frame(self)
        bottom_right_frame = tk.Frame(self)
        self.history_prompt = TextBox(bottom_left_frame, 'history', height=8, width=38)
        self.undo_button = tk.Button(
            bottom_left_frame,
            text="Undo last action",
            command=self.undo
            )
        self.reset_button = tk.Button(
            bottom_left_frame,
            text="Reset",
            command=self.reset
            )
        self.input_prompt = tk.Label(self, text="")
        self.square_data_prompt = TextBox(self, 'piece data', height=7, width=24)

        self.history_prompt.pack(side=tk.LEFT)
        self.undo_button.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.reset_button.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        bottom_left_frame.pack(side=tk.LEFT, anchor=tk.N+tk.W, expand=True)
        self.input_prompt.pack(side=tk.TOP, anchor=tk.N+tk.W, padx=20)
        self.square_data_prompt.pack(anchor=tk.N+tk.W)
        bottom_right_frame.pack(side=tk.RIGHT, anchor=tk.N+tk.W)

        # affiche l'état actuel de la partie
        self.square_data_prompt.clear()
        self.display()

    def reset(self):
        self.core.__init__()
        if self.fenstr:
            fenstrings.load(self.core, self.fenstr)
        self.core_interface.__init__(self.core)
        self.square_data_prompt.clear()
        self.display()

    def show_square_data(self, event):
        i, j = self.board.indexes_from_coords(event.x, event.y)
        try:
            sqr_id = square_id(i, j)
        except IndexError:
            pass
        else:
            sqr_data = self.core_interface.square_data(sqr_id)
            self.square_data_prompt.display_text(sqr_data)

    # inputs
    def click(self, event):
        i, j = self.board.indexes_from_coords(event.x, event.y)
        try:
            sqr_id = square_id(i, j)
        except IndexError:
            pass
        else:
            input_repr = self.core_interface.user_input(sqr_id)
            self.display(input_repr)
        self.show_square_data(event)

    def undo(self):
        self.core_interface.undo_action()
        self.display()


    # affichage
    def display_core_vars(self):
        """Affiche les valeurs de certaines variables du noyau"""
        txt = (
            f"{self.core._array_repr()}\n\n{self.core._info_repr()}\n\n"
            f"    win state\n{self.core_interface.get_win_state()}"
            )
        self.core_state.display_text(txt)

    def display_possibilities(self):
        """Affiche les actions possibles"""
        lines, bold_lines = choices_repr_lines(self.core_interface)
        self.possibilities.display_lines(lines, bold_lines)

    def display_board(self, marks:Iterable[int], mark_type:MarkType):
        """Affiche une représentation de l'échiquier du noyau"""
        self.board.display_squares()
        self.board.display_marks(marks, mark_type)
        self.board.display_marks(self.core_interface.search_all_errors(), MarkType.ERROR)

    def display(self, input_repr:str=None):
        if input_repr is None:
            self.input_prompt.config(
                text="--- current input ---\n",
                font=BOLD_FONT
                )
        else:
            self.input_prompt.config(
                text=f"--- current input ---\n{input_repr}",
                font=BOLD_FONT
                )
        self.history_prompt.display_lines(
            f"{i}: {action}" for i, action in zip(
                range(len(self.core.history), 0, -1),
                reversed(self.core.history)
                )
            )
        self.display_core_vars()
        self.display_possibilities()
        marks, mark_type = self.core_interface.marked_squares()
        self.display_board(marks, mark_type)


def start_debug_ui(*args):
    """
    start_debug_ui(core)
        Lance l'interface de debugage sur l'échiquier `core`

    start_debug_ui(core, fenstr)
        Charge la fenstring `fenstr` sur l'échiquier `core` et lance
        l'interface de debugage dessus

    start_debug_ui(fenstr)
        Lance l'interface de debugage sur un échiquier dont la fenstring est
        `fenstr`
    """
    if (
        len(args) == 2 and
        isinstance(args[0], chess.Chessgame) and
        isinstance(args[1], str)
        ):
        core, fenstr = args

    elif len(args) == 1:
        if isinstance(args[0], chess.Chessgame):
            core = args[0]
            fenstr = fenstrings.create(core)

        elif isinstance(args[0], str):
            fenstr = args[0]
            core = fenstrings.loaded(fenstr)

        else:
            raise TypeError

    else:
        raise TypeError

    game_viewer = GameViewer(core, fenstr)
    game_viewer.mainloop()


class BugTrackerFenstrings:
    position1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    position2 = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -"
    position3 = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -"
    position4 = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"
    position5 = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"


if __name__ == '__main__':
    default_start_fenstring = None
    promotion_test_fenstring = "rnbqkb1r/pppppp1p/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    no_pawn_fenstring = "rnbqkbnr/8/8/8/8/8/8/RNBQKBNR b KQkq - 0 1"
    castle_free_fenstring = "r3k2r/8/8/pppppppp/PPPPPPPP/8/8/R3K2R w KQkq - 0 1"
    bug_tracker_fenstring = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"

    RUN_IN_BACKGROUND = False
    CORE = chess.Chessgame()
    FENSTRING = BugTrackerFenstrings.position4
    # FENSTRING = "3r2n1/1bN4r/R1pbBp1p/1p2p1p1/1Pk1P1P1/2P5/1Q2K2P/1NBR4 b - - 0 1"

    if RUN_IN_BACKGROUND:
        thread = threading.Thread(target=start_debug_ui, args=(FENSTRING,))
        thread.start()
    else:
        start_debug_ui(FENSTRING)