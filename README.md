Importation des modules
--------------------------------------------------------------------------------
```py
import chess       # pour simuler une partie d'echecs
import fenstrings  # pour pouvoir charger des états depuis des fenstrings
```



Créer un échiquier
--------------------------------------------------------------------------------
```py
core = chess.Chessgame()
```

Une instance de `chess.Chessgame` est un objet qui simule l'état d'une partie d'échec et fournit des méthodes permettant d'explorer son arbre des possibilités.
Par défaut, un noyau est créé dans la position initiale d'une nouvelle partie d'échec (fenstring correspondante : `"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"`).

Après avoir créé le noyau, il est possible de le mettre dans l'état décrit par
la fenstring `fen`` :
```py
fenstrings.load(core, fen)
```

Ou bien il est aussi possible de créer un noyau directement initialisé dans
l'état décrit par la fenstring fen :
```py
core = fenstrings.loaded(fen)
```

Exemples de fenstrings intéressantes :
- `position_initiale = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"`
- `position2 = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -"`
- `position3 = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -"`
- `position4 = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"`
    


La modélisation des pièces de l'échiquier
--------------------------------------------------------------------------------
Les pièces du jeu d'échec sont représentées par des objets dont les types
héritent de la classe `chess.Piece`

Une pièce possède (entre autre) les attributs suivants :
- `square : int`
l'identifiant de la case sur laquelle la pièce est positionnée dans
l'état actuel de la partie

- `colour : bool`
la couleur de la pièce
`False` => pièce noire
`True` => pièce blanche

- `alive : bool`
`True` => la pièce est encore en jeu (i.e elle n'a pas été capturée)
`False` => la pièce est hors-jeu (i.e elle a été capturée)

En plus des pièces, il existe les deux singletons `chess.EMPTY` et `chess.OUT`
La classe `chess.Piece` et les classes des singletons `chess.EMPTY` et `chess.OUT`
héritent toutes de la classe `chess.ChessboardItem`



La modélisation d'une partie d'échec
--------------------------------------------------------------------------------
Une partie d'échec peut être formellement vue comme une machine d'état dont
les variables sont :
- `player : `bool
Couleur du joueur qui est en train d'effectuer une action
`chess.WHITE` == `True` == joueur blanc
`chess.BLACK` == `False` == joueur noir

- `array : List[ChessboardItem]`
Un tableau d'instances de `chess.ChessboardItem` à une dimension qui
représente le contenu de l'échiquier à l'état actuel.
Chaque case de ce tableau peut être :
    - Une instance de `chess.Piece`
    - `chess.EMPTY` : ce singleton représente une case vide de l'échiquier
    (aucune pièce n'est posée sur cette case à l'état actuel, mais il
    existe d'autres états de la partie d'échecs où une pièce est posée
    dessus)
    - `chess.OUT` : ce singleton représente l'extérieur de l'échiquier
    (c'est un endroit inaccessible aux pièces : parmi tous les états
    possibles de la partie d'échec, il n'en existe aucun où une pièce
    est posée à cet endroit)

- `won_material : Tuple[List[Piece]]`
`board.won_material[colour]` == liste des pièces que le joueur `colour` a
capturé à son adversaire

- `pieces : Tuple[List[Piece]]`
`core.pieces[colour]` == liste de toutes les pièces de la couleur `colour`

- `rook_castling : List[List[bool]]`
`core.rook_castling[colour][side]` == `True` ssi la tour de la couleur
`colour` et du côté `side` sur l'échiquier `board` ne s'est encore jamais
déplacée

- `en_passant : Union[None, int]`
Si une prise en passant est possible, alors `core.en_passant` est la
position d'arrivée du pion qui effectue la prise en passant.
Sinon, `en_passant` vaut None

On considère que la partie change d'état dès qu'au moins l'une de ces variables change de valeur.
Dans les faits, un changement d'état se repère par l'inversion de la valeur du booléen `player`.
L'application qui, à un état de la partie en associe un autre, est appelée une action.
Dans ce programme, les actions sont donc des objets qui, selon leur type, peuvent modifier d'une certaine façon les variables de la machine d'états à laquelle ils sont liés.



La modélisation des actions de l'échiquier
--------------------------------------------------------------------------------
Les actions du jeu d'échec sont représentées par des objets dont les types
héritent de la classe `chess.Action`

Il existe plusieurs types d'actions :
- `chess.Movement` : Déplacement classique d'une pièce quelconque d'une case vers une autre, pouvant éventuellement capturer une pièce adverse
- `chess.Promotion` : Promotion d'un pion en une autre pièce
- `chess.LongPush` : Déplacement d'un pion de deux cases en ligne droite vers le côté adverse
- `chess.EnPassant` : Prise en passant d'un pion
- `chess.Castle` : Roque


Selon leurs types, les actions possèdent des attributs différents. Mais
elles possèdent toutes au moins les deux attributs suivants :
- `board : chess.Chessgame`
L'échiquier sur lequel l'action peut être effectuée (c'est-à-dire la machine d'états à laquelle est liée l'objet action)

- `piece : chess.piece`
La pièce de l'échiquier qui effectue l'action (par exemples : le pion qui effectue une prise en passant, le roi qui effectue un roque, ou le fou qui effectue un déplacement classique)

Les actions possèdent des méthodes qui permettent de manipuler les variables
de la machine d'états présentées plus haut :
- `do()` : Si l'action ne met ni ne laisse le roi du joueur en échec, alors effectue cette action et renvoie `True`.
Sinon, ne fait rien en renvoie `False`.
    
- `undo()` :
Annule l'action à condition que celle-ci ait déjà été effectuée avec sa méthode `.do()` et qu'aucune autre action n'ait été effectuée après cela.
Modifie les variables de l'échiquier pour le remettre dans l'état dans lequel il était avant l'exécution de cette action.


À titre indicatif :
Fonctionnement de `action.do()` :
```
Appelle action._execute() qui commence à effectuer l'action en ne modifiant
que la valeur de la variable array de l'échiquier.

Si le tableau array décrit une position de l'échiquier dans laquelle le
roi du joueur qui effectue l'action est en échec, alors
    execute action._clean() qui rétablit la valeur de la variable array,
    puis renvoie False pour indiquer que cette action n'est pas légale.
Sinon,
    execute action._finalize() qui modifie les autres variables de
    l'échiquier afin de totalement le faire basculer dans son état suivant,
    puis renvoie True pour indiquer que l'action est légale et qu'elle
    a bien été effectuée sur l'échiquier.
```



Utilisation du module par un moteur d'échec pour générer les actions possibles
--------------------------------------------------------------------------------
`core.possibilities()` permet d'itérer sur toutes les actions possibles à effectuer depuis l'état actuel de l'échiquier.
Cependant, pour essayer d'être plus performante, cette méthode limite le plus possible les modifications des variables de l'échiquier.
La méthode ne vérifie donc pas si les actions qu'elle génère ne laissent ni ne mettent le roi du joueur en échec.
Cette vérification doit donc être faite à la volée par le moteur d'échec lorsqu'il explore l'arbre des possibilités.
Ainsi, dans le cas où la méthode `.do()` d'une action renvoie `False`, alors cela signifie qu'elle n'est pas légale et qu'il est donc inutile d'explorer toutes les réponses que l'adversaire pourrait faire suite à cette action.



Utilisation du module pour réaliser une interface graphique
--------------------------------------------------------------------------------
Tout d'abord, l'objet `core` repère chaque case de l'échiquier avec un
identifiant qui est un nombre entier positif unique compris entre 0 et 77.
Quand on observe l'échiquier avec le côté des noirs en haut et celui des blancs
en bas, son chiffre de la dizaine est l'indice de la ligne de la case et son
chiffre de l'unité est l'indice de la colonne de la case :

```
  |A |B |C |D |E |F |G |H |
8 | 0| 1| 2| 3| 4| 5| 6| 7| (0-7)         |A8|B8|C8|D8|E1|F8|G8|H8|
7 |10|11|12|13|14|15|16|17| (10-17)       |A7|B7|C7|D7|E2|F7|G7|H7|
6 |20|21|22|23|24|25|26|27| (20-27)       |A6|B6|C6|D6|E3|F6|G6|H6|
5 |30|31|32|33|34|35|36|37| (30-37)  <=>  |A5|B5|C5|D5|E4|F5|G5|H5|
4 |40|41|42|43|44|45|46|47| (40-47)       |A4|B4|C4|D4|E5|F4|G4|H4|
3 |50|51|52|53|54|55|56|57| (50-57)       |A3|B3|C3|D3|E6|F3|G3|H3|
2 |60|61|62|63|64|65|66|67| (60-67)       |A2|B2|C2|D2|E7|F2|G2|H2|
1 |70|71|72|73|74|75|76|77| (70-77)       |A1|B1|C1|D1|E8|F1|G1|H1|
   (identifiants des cases)                    (noms des cases)
```

Pour générer l'ensemble de toutes les actions que peut faire le joueur,
l'interface graphique peut utiliser la méthode `.choices()` du noyau.

`core.choices()` analyse l'état actuel de l'échiquier et renvoie un tuple de la
forme `(win_state, choices)`

- `win_state` décrit si la partie est terminée et de quelle manière. `win_state` peut prendre les valeurs suivantes :
    - `chess.NOT_FINISHED` : La partie n'est pas terminée
    - `chess.MAT` : Il y a échec et mat. La couleur du perdant est `chess.player`, celle du gagnant est `not chess.player`
    - `chess.NULL_PAT` : Il y a match nul par pat
    - `chess.NULL_REPETITIONS` : Il y a match nul à cause d'un certain nombre de répétitions successives du même enchainement d'états (Cette vérification n'est pas encore implémenté)

- `choices` est un dictionnaire de dictionnaires qui fait correspondre les inputs du joueur sur interface graphique avec les actions possibles à effectuer sur l'échiquier.
Si une pièce susceptible d'effectuer une action est présente sur la case dont l'identifiant est `start`, alors `choices[start]` existe et est un dictionnaire.
Dans ce cas, si cette même pièce peut se déplacer vers la case dont l'identifiant est `end`, alors `choices[start][end]` est l'instance de `chess.Action` qui permet d'effectuer l'action déplaçant la pièce de `start` vers `end` sur
l'échiquier `core`.


Pseudo code pour effectuer une action sur l'échiquier `core` via une interface graphique :

```
choices = core.choices()
for sqr_id in choices.keys() :
    < mettre en surbrillance la case dont l'identfiant est sqr_id >

< l'utilisateur selectionne la case dont l'identifiant est start >
actions = choices.get(start)
if actions is None :
    < La case start ne contient pas de pièce pouvant se déplacer >
else :
    for end in actions.keys() :
        < mettre en surbrillance la case dont l'identifiant est end >
        
< l'utilisateur selectionne la case dont l'identifiant est end >
act = actions.get(end)
if act is None :
    < la pièce séléctionnée ne peut pas se déplacer vers la case end >
else :
    if core[end] is not chess.EMPTY : # capture d'une pièce adverse
        < affiche une représentation graphique de la pièce core[end] dans le
          materiel capturé par le joueur >
    
    if isinstance(act, chess.Promotion) : # promotion d'un pion
        piece_type = < demander à  l'utiliseur en quel type de pièce il veut
                       transformer son pion >
        act.promote_into(piece_type)
    
    < afficher l'animation de l'action act sur l'interface graphique >
    act.do() # effectue l'action act sur l'échiquier core
```


Pseudo code pour annuler une action via une interface graphique :

```
act = core.undo() # annule la dernière action effectuée sur l'échiquier core
if act is None :
    < Il n'y a aucune action à annuler >
else :
    # Affiche l'animation d'annulation :
    
    if isinstance(act, chess.Promotion) :
        < afficher l'animation de l'annulation de la promotion >
        # On peut obtenir la case de départ du pion avant sa promotion avec
        # start_square(act).
        
        # Si une pièce a été capturée lors de cette promotion, alors on peut
        # l'obtenir avec captured_piece(act). La case sur laquelle il faut
        # afficher à nouveau sa représentation graphique est end. Et sa couleur
        # est enemy_colour(act) (qui vaut chess.WHITE ou chess.BLACK).
        
        # Si aucune pièce n'a été capturée, captured_piece(act) vaut
        # chess.EMPTY (et sa valeur de vérité est False).
    
    
    elif isinstance(act, (chess.Movement, chess.LongPush)) :
        < afficher l'animation de l'annulation de l'action act >
        # On peut obtenir la case de départ de la pièce avec start_square(act).
        
        # Si une pièce a été capturée lors de cette action, alors on peut
        # l'obtenir avec captured_piece(act). La case sur laquelle il faut
        # afficher à nouveau sa représentation graphique est end. Et sa couleur
        # est enemy_colour(act) (qui vaut chess.WHITE ou chess.BLACK).
        
        # Si aucune pièce n'a été capturée, captured_piece(act) vaut
        # chess.EMPTY (et sa valeur de vérité est False).
    
    
    elif isinstance(act, chess.EnPassant) :
        < afficher l'animation de l'annulation de la prise en passant >
        # La case sur laquelle il faut afficher à nouveau la représentation
        # graphique du pion capturé est captured_square(act) et la couleur de
        # ce pion est
        # enemy_colour(act) (qui vaut chess.WHITE ou chess.BLACK).
    
    
    elif isinstance(act, chess.Castle) :
        < afficher l'animation de l'annulation du roque >
        # castle_side(act) vaut chess.LONG s'il s'agit du grand roque et
        # chess.SHORT s'il s'agit du petit roque.

        # action_colour(act) vaut chess.WHITE s'il s'agit d'un roque des blancs
        # et chess.BLACK s'il s'agit d'un roque des noirs.
```
