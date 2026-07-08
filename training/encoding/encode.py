import chess

PIECE_TYPE_ORDER = "pnbrq"
THREAT_PIECE_TYPE_ORDER = "pnbrqk"

NUM_KING_BUCKETS = 32
NUM_PIECE_TYPE_SLOTS = 11  # own p/n/b/r/q, opp p/n/b/r/q, opp king
SLOTS_PER_BUCKET = NUM_PIECE_TYPE_SLOTS * 64
NUM_PLACEMENT_FEATURES = NUM_KING_BUCKETS * SLOTS_PER_BUCKET

NUM_THREAT_PIECE_TYPES = 6  # p,n,b,r,q,k
NUM_THREAT_RELATIONS = 2  # own-attacks-enemy, enemy-attacks-own
THREATS_PER_BUCKET = NUM_THREAT_RELATIONS * NUM_THREAT_PIECE_TYPES * NUM_THREAT_PIECE_TYPES
NUM_THREAT_FEATURES = NUM_KING_BUCKETS * THREATS_PER_BUCKET

NUM_FEATURES = NUM_PLACEMENT_FEATURES + NUM_THREAT_FEATURES


def parse_fen_placement(fen):
    placement = fen.split(" ")[0]
    board = [[" "] * 8 for _ in range(8)]
    r, c = 0, 0
    for ch in placement:
        if ch == "/":
            r += 1
            c = 0
        elif ch.isdigit():
            c += int(ch)
        else:
            board[r][c] = ch
            c += 1
    return board


def _transform(r, c, is_black, mirror):
    if is_black:
        r = 7 - r
    if mirror:
        c = 7 - c
    return r, c


def _find_king(board, want_white):
    for r in range(8):
        for c in range(8):
            p = board[r][c]
            if p == " ":
                continue
            if p.upper() == "K" and p.isupper() == want_white:
                return r, c
    raise ValueError("king not found")


def _king_bucket(board, persp_is_white):
    kr0, kc0 = _find_king(board, persp_is_white)
    kr0, kc0 = _transform(kr0, kc0, not persp_is_white, mirror=False)
    mirror = kc0 >= 4
    if mirror:
        kc0 = 7 - kc0
    return kr0 * 4 + kc0, mirror


def perspective_features(board, persp_is_white):
    king_bucket, mirror = _king_bucket(board, persp_is_white)

    indices = []
    for r in range(8):
        for c in range(8):
            p = board[r][c]
            if p == " ":
                continue
            is_white_piece = p.isupper()
            is_own = is_white_piece == persp_is_white
            is_king = p.upper() == "K"

            if is_king:
                if is_own:
                    continue  # own king is the anchor, never a placed feature
                piece_type_idx = 10  # opponent king, shared slot
            else:
                type_idx = PIECE_TYPE_ORDER.index(p.lower())
                piece_type_idx = type_idx if is_own else 5 + type_idx

            tr, tc = _transform(r, c, not persp_is_white, mirror)
            sq_idx = tr * 8 + tc
            indices.append(king_bucket * SLOTS_PER_BUCKET + piece_type_idx * 64 + sq_idx)
    return indices


def _compute_threat_pairs(fen):
    board = chess.Board(fen)
    pairs = set()
    for from_sq in chess.SQUARES:
        attacker = board.piece_at(from_sq)
        if attacker is None:
            continue
        for to_sq in board.attacks(from_sq):
            victim = board.piece_at(to_sq)
            if victim is None or victim.color == attacker.color:
                continue
            attacker_type = THREAT_PIECE_TYPE_ORDER.index(chess.piece_symbol(attacker.piece_type))
            victim_type = THREAT_PIECE_TYPE_ORDER.index(chess.piece_symbol(victim.piece_type))
            pairs.add((attacker.color == chess.WHITE, attacker_type,
                       victim.color == chess.WHITE, victim_type))
    return pairs


def perspective_threat_features(threat_pairs, board, persp_is_white):
    king_bucket, _ = _king_bucket(board, persp_is_white)
    seen = set()
    indices = []
    for attacker_is_white, attacker_type, victim_is_white, victim_type in threat_pairs:
        attacker_is_own = attacker_is_white == persp_is_white
        relation = 0 if attacker_is_own else 1
        key = (relation, attacker_type, victim_type)
        if key in seen:
            continue
        seen.add(key)
        idx = (NUM_PLACEMENT_FEATURES + king_bucket * THREATS_PER_BUCKET +
               relation * NUM_THREAT_PIECE_TYPES * NUM_THREAT_PIECE_TYPES +
               attacker_type * NUM_THREAT_PIECE_TYPES + victim_type)
        indices.append(idx)
    return indices


def fen_to_feature_indices(fen):
    board = parse_fen_placement(fen)
    stm_is_white = fen.split(" ")[1] == "w"

    stm_indices = perspective_features(board, stm_is_white)
    ntm_indices = perspective_features(board, not stm_is_white)

    threat_pairs = _compute_threat_pairs(fen)
    stm_indices += perspective_threat_features(threat_pairs, board, stm_is_white)
    ntm_indices += perspective_threat_features(threat_pairs, board, not stm_is_white)

    return stm_indices, ntm_indices


NUM_OUTPUT_BUCKETS = 8


def output_bucket_for_board(board):
    piece_count = sum(1 for row in board for p in row if p != " ")
    return min((piece_count - 1) // 4, NUM_OUTPUT_BUCKETS - 1)


def fen_to_output_bucket(fen):
    return output_bucket_for_board(parse_fen_placement(fen))
