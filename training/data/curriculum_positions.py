import random

import chess

PIECE_TYPES_NO_KING = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]


def selfplay_fens(rng, num_games, plies_per_game, sample_every=3):
    fens = []
    for _ in range(num_games):
        board = chess.Board()
        for ply in range(plies_per_game):
            if board.is_game_over():
                break
            legal = list(board.legal_moves)
            if not legal:
                break
            if ply % sample_every == 0 and not board.is_check():
                fens.append(board.fen())
            move = legal[rng.randrange(len(legal))]
            board.push(move)
    return fens


def post_capture_fens(rng, num_games, plies_per_game):
    fens = []
    for _ in range(num_games):
        board = chess.Board()
        for _ in range(plies_per_game):
            if board.is_game_over():
                break
            legal = list(board.legal_moves)
            if not legal:
                break
            move = legal[rng.randrange(len(legal))]
            was_capture = board.is_capture(move)
            board.push(move)
            if was_capture and not board.is_check():
                fens.append(board.fen())
    return fens


def material_imbalance_fens(rng, num_positions, max_random_plies=14):
    fens = []
    attempts = 0
    while len(fens) < num_positions and attempts < num_positions * 5:
        attempts += 1
        board = chess.Board()
        random_plies = 4 + rng.randrange(max_random_plies)
        ok = True
        for _ in range(random_plies):
            legal = list(board.legal_moves)
            if not legal:
                ok = False
                break
            board.push(legal[rng.randrange(len(legal))])
        if not ok or board.is_game_over():
            continue

        target_color = rng.choice([chess.WHITE, chess.BLACK])
        candidates = [
            sq for sq in chess.SQUARES
            if board.piece_at(sq) is not None
            and board.piece_at(sq).piece_type != chess.KING
            and board.piece_at(sq).color == target_color
        ]
        if not candidates:
            continue
        remove_count = min(len(candidates), 1 + rng.randrange(3))
        rng.shuffle(candidates)
        for sq in candidates[:remove_count]:
            board.remove_piece_at(sq)

        if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
            continue
        board.turn = rng.choice([chess.WHITE, chess.BLACK])
        if not board.is_valid():
            continue
        fens.append(board.fen())
    return fens


def _random_square(rng, exclude):
    while True:
        sq = rng.randrange(64)
        if sq not in exclude:
            return sq


def _kings_not_adjacent(k1, k2):
    return chess.square_distance(k1, k2) > 1


def endgame_fens(rng, num_positions):
    skeletons = ["KR", "KQ", "KP", "KBN", "KRR", "KRP"]
    fens = []
    attempts = 0
    while len(fens) < num_positions and attempts < num_positions * 6:
        attempts += 1
        skeleton = skeletons[rng.randrange(len(skeletons))]
        strong_side = rng.choice([chess.WHITE, chess.BLACK])
        weak_side = not strong_side

        board = chess.Board.empty()
        used = set()
        wk = _random_square(rng, used)
        used.add(wk)
        bk = _random_square(rng, used)
        while not _kings_not_adjacent(wk, bk):
            bk = _random_square(rng, used)
        used.add(bk)

        board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))

        extra_types = {
            "KR": [chess.ROOK], "KQ": [chess.QUEEN], "KP": [chess.PAWN],
            "KBN": [chess.BISHOP, chess.KNIGHT], "KRR": [chess.ROOK, chess.ROOK], "KRP": [chess.ROOK, chess.PAWN],
        }[skeleton]

        ok = True
        for pt in extra_types:
            sq = _random_square(rng, used)
            tries = 0
            while pt == chess.PAWN and (chess.square_rank(sq) == 0 or chess.square_rank(sq) == 7) and tries < 20:
                sq = _random_square(rng, used)
                tries += 1
            if pt == chess.PAWN and (chess.square_rank(sq) == 0 or chess.square_rank(sq) == 7):
                ok = False
                break
            used.add(sq)
            board.set_piece_at(sq, chess.Piece(pt, chess.WHITE if strong_side == chess.WHITE else chess.BLACK))
        if not ok:
            continue

        board.turn = rng.choice([chess.WHITE, chess.BLACK])
        if not board.is_valid():
            continue
        if board.is_checkmate() or board.is_stalemate():
            continue
        fens.append(board.fen())
    return fens


def build_curriculum(seed, selfplay_games=400, plies_per_game=60, material_positions=6000, endgame_positions=6000):
    rng = random.Random(seed)
    fens = []
    fens.extend(selfplay_fens(rng, selfplay_games, plies_per_game))
    fens.extend(post_capture_fens(rng, selfplay_games // 2, plies_per_game))
    fens.extend(material_imbalance_fens(rng, material_positions))
    fens.extend(endgame_fens(rng, endgame_positions))
    rng.shuffle(fens)
    return fens
