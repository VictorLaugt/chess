#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 23:29:25 2023

@author: victor
"""

import tkinter as tk
from PIL import Image, ImageTk

from typing import Iterable, Tuple, List, Dict

import chess
import fenstrings


# ---- Back
class ChessGameControler :
    """Fait l'interface entre le front (l'interface graphique) et le back
    (l'échiquier de type chess.Chessboard).
    """
    def __init__(self, game:chess.Chessgame) :
        self._game = game
        self._selected_sqr = None
        self._asc_hist : List[chess.Action] = []
        self._init_game_state()

    def _init_game_state(self) :
        winstate, choices = self._game.choices()
        self._choices : Dict[int, Dict[int, chess.Action]] = choices
        self._winstate : chess.WinState = winstate
        self._selected_start_sqr : int = None

    def _do_action(self, action:chess.Action) :
        self._asc_hist.clear()
        action.do()
        self._winstate, self._choices = self._game.choices()

    def _select_start_sqr(self, start_sqr:int) -> Iterable[int] :
        end_sqr_choices = self._choices.get(start_sqr)
        if end_sqr_choices is not None :
            self._selected_start_sqr = start_sqr
            return end_sqr_choices.keys()
        return ()

    def _select_end_sqr(self, end_sqr:int) :
        action = self._choices[self._selected_start_sqr].get(end_sqr)
        if action is not None :
            self._do_action(action)
        self._selected_start_sqr = None

    @property
    def current_player(self) :
        return self._game.player

    @property
    def winstate(self) :
        return self._winstate

    def select_sqr(self, sqr:int) -> Iterable[int] :
        if self._selected_start_sqr is None :
            return self._select_start_sqr(sqr)

        else :
            self._select_end_sqr(sqr)
            return ()

    def undo(self) -> bool :
        undone_action = self._game.undo()
        if undone_action is None :
            return False
        self._asc_hist.append(undone_action)
        return True

    def redo(self) -> bool :
        if not self._asc_hist :
            return False
        self._asc_hist.pop().do()
        return True

    def load_game(self, fen:str=None) :
        fenstrings.load(self._game, fen)
        self._init_game_state()


# ---- Front
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
    chess.Pawn : "/pion.png",
    chess.Bishop : "/fou.png",
    chess.Knight : "/cavalier.png",
    chess.Rook : "/tour.png",
    chess.Queen : "/dame.png",
    chess.King : "/roi.png",
    }


# TODO
def coords_to_square_id(x:int, y:int) -> int :
    ...

def indexes_to_square_id(i:int, j:int) -> int :
    ...

def square_id_to_indexes(sqr:int) -> Tuple[int] :
    ...

def square_id_to_square_name(sqr:int) -> str :
    ...


class Sprite(ImageTk.PhotoImage) :
    def __init__(self, path:str, width:int) :
        img = Image.open(path)
        l_img = img.size[0]
        if l_img != width :
            img = img.resize((width, width))
        super().__init__(img)


def piece_sprite(piece_type:type, colour:bool, width:int) -> Sprite :
    """Crée et renvoie un Sprite de largeur width représentant une pièce de
    type piece_type et de la couleur colour
    """
    pieces_dir = "/_effective_pieces"
    if colour :
        colour_dir = "/_white"
    else :
        colour_dir = "/_black"
    path = IMAGE_PATH + pieces_dir + colour_dir + IMAGE_NAMES[piece_type]
    return Sprite(path, width)


class Board(tk.Canvas) :
    """Canvas qui permet d'afficher un échiquier"""
    def __init__(self, master) :
        super().__init__(master)
        board_size = self.sqr_size * 9
        self.configure(width=board_size, height=board_size, bg=MAIN_BLUE)

        self.init_sprites()
        self.display_edges()

    @property
    def sqr_size(self) -> int :
        """Longueur du coté de chaque case de l'échiquier"""
        # height = self.master.winfo_height()
        # return round(height/10)
        return 70

    @property
    def coeff(self) -> float :
        return self.sqr_size / 80

    @property
    def marge(self) -> int :
        return round(30*self.coeff)

    def init_sprites(self) :
        """Crée les sprites à afficher sur l'échiquier"""
        sprite_size = self.sqr_size
        self.mark_sprite = Sprite(
            IMAGE_PATH + "/green_highlight.png",
            sprite_size
            )
        self.piece_sprite = {}
        for ptype in PIECE_TYPES :
            self.piece_sprite[ptype] = (
                piece_sprite(ptype, BLACK, sprite_size), # sprite noir
                piece_sprite(ptype, WHITE, sprite_size)  # sprite blanc
                )

    def get_piece_sprite(self, piece_type:type, colour:bool) -> Sprite :
        """Sprite de la pièce du type piece_type et de la couleur colour"""
        return self.piece_sprite[piece_type][colour]

    def indexes_to_coords(self, i:int, j:int) -> Tuple[int] :
        """board.indexes_to_coords(i, j) == coordonnées (x, y) du sprite
        représentant la pièce située à la case d'indices (i, j) sur l'échiquier
        board
        """
        marge = self.marge
        return (j*self.sqr_size + 1.5*marge, i*self.sqr_size + 1.5*marge)

    def display_edges(self) :
        """Afiche les bordures de l'échiquier"""
        font_size = str(round(18*self.coeff))
        marge = self.marge

        # Affichage des numéros des cases
        for i in range(8) :
            y = (i+1) * self.sqr_size
            number = str(8-i)
            for (x, anch) in ((9*self.sqr_size-marge, tk.W), (marge, tk.E)) :
                self.create_text(
                    x,
                    y,
                    text=number,
                    anchor=anch,
                    font=(FONT, font_size),
                    fill=WSQR_COLOR,
                    tags="edge_number"
                    )

        # Affichage des lettres des cases :
        for i, letter in zip(range(8), "ABCDEFGH") :
            x = (i+1) * self.sqr_size
            for (y, anch) in ((9*self.sqr_size-marge, tk.N), (marge, tk.S)) :
                self.create_text(
                    x,
                    y,
                    text=letter,
                    anchor=anch,
                    font=(FONT, font_size),
                    fill=WSQR_COLOR,
                    tags="edge_letter"
                    )

    def display_squares(self) :
        """Recharge l'affiche les cases"""
        self.delete("square")
        for i in range(8) :
            for j in range(8) :
                if (i+j) % 2 == 0 :
                    sqr_colour = WSQR_COLOR
                else :
                    sqr_colour = BSQR_COLOR
                rectangle_points = (
                    *self.indexes_to_coords(i, j),
                    *self.indexes_to_coords(i+1, j+1)
                    )
                self.create_rectangle(
                    rectangle_points,
                    fill=sqr_colour,
                    width=0,
                    tags=("square", f"s{i}{j}", f"{sqr_colour}")
                    )

    def display_pieces(self) :
        """Recharge l'affichage des pièces"""
        self.delete("piece")
        for i in range(8) :
            for j in range(8) :
                x, y = self.indexes_to_coords(i, j)
                piece = self.game[indexes_to_square_id(i, j)]
                if piece :
                    sprite = self.get_piece_sprite(type(piece), piece.colour)
                    self.create_image(
                        x,
                        y,
                        image=sprite,
                        anchor=tk.NW,
                        tags=("piece", f"p{i}{j}")
                        )
        self.tag_raise("piece")

    def display_marks(self, marks:Iterable[int]) :
        """Recharge l'affichage des marques"""
        for sqr in marks :
            i, j = square_id_to_indexes(sqr)
            x, y = self.indexes_to_coords(i, j)
            self.create_image(
                x,
                y,
                image=self.mark_sprite,
                anchor=tk.NW,
                tags=("mark", f"m{i}{j}")
                )
        self.tag_raise("mark")

    def move_piece(self) : # TODO
        ...