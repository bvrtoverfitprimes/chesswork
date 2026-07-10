# Classical eval/search gaps — what strong pre-NNUE engines have that we don't

Sources drawn on directly: Stockfish 11 `evaluate.cpp` (the last full HCE), Ethereal
12/13 `evaluate.c`, Berserk 4–8 `eval.c`/`endgame.c`, Weiss, chessprogramming.org.
Our current eval/search inventory is in WORKLOG §34; this lists only MISSING items.

## Ranked top-10 by (documented Elo) / (implementation cost) for an engine at ~2400

1. **Automated Texel/gradient tuning of every weight** — biggest single lever, bar none.
   All our weights are hand-guessed. Tuning an untuned-but-complete HCE on quiet labeled
   positions is routinely +100–250 Elo (Ethereal, Weiss, Berserk devlogs). Cost: medium
   (need data + tuner), but pays back more than all terms below combined. DO THIS.
2. **King-safety "safe checks"** — the dominant king-safety component in SF/Ethereal/Berserk,
   entirely absent from ours. Count squares from which each enemy piece type could give
   check that are *safe* (undefended by us / we control). Weighted huge. ~+30–60 Elo. Cost: med.
3. **Endgame scale factors** (opposite-colored bishops, drawish material, 50-move ramp) —
   cheap post-hoc multiplier on the eg score. Stops us converting won-looking OCB/KRPKR into
   thrown half-points and stops overvaluing dead draws. ~+20–40 Elo. Cost: LOW. DO FIRST.
4. **Threats, granular** (threat-by-minor / threat-by-rook per victim type, threat-by-safe-pawn,
   pawn-push threat, restricted squares, weak-queen-protection). Our threats are 2 crude terms.
   SF/Berserk devote ~10 tuned values here. ~+20–40 Elo. Cost: med.
5. **Connected / phalanx / supported pawns** (Connected[rank] bonus, big). Plus backward pawns.
   Our pawn eval is only doubled+isolated+passed. ~+15–30 Elo. Cost: low-med (pawn hash already
   exists).
6. **King-danger nonlinear + weak-ring-squares + no-queen** — upgrade our attack-unit sum into
   the real `kingDanger` accumulator (attackerWeight, attacksOnRing, weak squares = ring squares
   attacked by enemy & defended ≤ once, flank/king-mobility, no-enemy-queen bonus), then
   `score -= danger*danger / 4096`. ~+20 Elo over our crude table. Cost: med (folds into #2).
7. **Rook behind passed pawn (Tarrasch) + rook on 7th/relative-7th** — big in R endings, cheap.
   ~+10–20 Elo. Cost: LOW.
8. **Mobility as tuned per-count tables** (not a linear `(mob - ref)*w`). Every strong engine
   uses `MobilityBonus[pieceType][count]` tuned tables, mg+eg. Also fixes our current linear
   slope being a guess. Folds into the tuning project (#1). ~+10–20 Elo. Cost: low w/ tuner.
9. **Space term** (SF: safe center-file squares in own half behind pawns, ×(pieceCount + blocked)).
   Matters most vs a cramped opponent; opening/middlegame. ~+10 Elo. Cost: low.
10. **Bad bishop / minor-behind-pawn / bishop-on-long-diagonal / trapped-bishop-a2a7h2h7** —
    classic minor-piece nuances. ~+10 Elo combined. Cost: low. (Note: outposts we already tried
    in a bundle; retry individually AFTER tuning so weights are real.)

Deferred / low-priority for us: full imbalance table (SF `Imbalance` — complex, ~+10, high cost);
Syzygy TB (real but a whole integration; endgame scale factors get most of the practical benefit
cheaply); pawn-storm files (folds into #6).

## Concrete formulas

### #3 Endgame scale factors (IMPLEMENT FIRST — safe, cheap)
After computing tapered `score`, before returning, scale the eg contribution:
```
scale = 128  # /128 = 1.0
winner = side with score>0
# opposite-colored bishops, no other pieces except pawns:
if each side has exactly 1 bishop AND bishops are opposite colors:
    if no knights/rooks/queens: scale = 64 + 2*passedCount(winner)   # OCB pure = very drawish
    else:                       scale = 96                            # OCB with other pieces
# few pawns for the winning side => harder to win:
if winnerPawns == 0 and winnerNonPawnMaterial <= bishopValue: scale = min(scale, 32)  # KBK etc
if winnerPawns == 1: scale = min(scale, 96)
# 50-move ramp:
scale = scale * (100 - halfmoveClock) / 100   # if clock available; else skip
score = (score * scale) / 128    # apply only to the eg-heavy portion, or whole score in low phase
```
Berserk/SF apply scale to the endgame score component; simplest robust version: apply to whole
`score` when `phase` is low (endgame). Verify OCB pure ending scores ~half of same material SCB.

### #2 + #6 King safety with safe checks (the big one)
Requires a first pass that fills, for BOTH sides: `attackedBy[side]` (have it),
`attackedBy2[side]` (≥2 attackers), and per-type union `pieceAtt[side][knight|bishop|rook|queen]`.
Then per defended king (king of `us`, attacker = `them`):
```
ring   = kingRing[us]
weak   = ring & attackedBy[them] & ~attackedBy2[us]        # ring squares they attack, we defend ≤once
danger = attackerWeight_sum(them)                          # Σ attackerCnt*typeWeight (N81 B52 R44 Q10 style)
       + 69 * attacksOnRingCount(them)
       + 185 * popcount(weak & ring)
       - 100 * (no we-have-knight-defender on ring)         # optional
# safe squares for them to land a checking piece:
safe = ~own[them] & (~attackedBy[us] | (weak & attackedBy2[them]))
rookChk   = rookAttacks(ksq, occ)     & pieceAtt[them][rook]
bishopChk = bishopAttacks(ksq, occ)   & pieceAtt[them][bishop]
knightChk = knightAttacks[ksq]        & pieceAtt[them][knight]
queenChk  = (rookAttacks|bishopAttacks)(ksq,occ) & pieceAtt[them][queen]
danger += 1080 * popcount(rookChk   & safe) ? use per-type weights:  RookSafeCheck 1080
        + 780  * popcount(queenChk  & safe)     QueenSafeCheck 780
        + 790  * popcount(bishopChk & safe)     BishopSafeCheck 790
        + 640  * popcount(knightChk & safe)     KnightSafeCheck 640   (SF11 units)
        + (smaller) * unsafe checks
danger -= 873 * (them has no queen)                          # NoQueen
kingSafetyScore(us) = danger>0 ? danger*danger/4096 : 0      # nonlinear; subtract from us's score
```
Weights above are SF11 order-of-magnitude in "internal units"; for OUR cp scale, divide the final
`danger*danger/4096` result by ~a constant (tune). Simplest first cut that will already beat our
crude table: keep our attack-unit sum, ADD `+800*safeCheckCount` into the units before the table
lookup, and widen the table. Verify: a position with a safe queen check near the king should now
swing 150–300cp vs before.

### #4 Threats (granular)
```
# them's pieces we attack:
theirNonPawn = own[them] & ~pawns[them] & ~king[them]
# threat by our minor:
t = theirNonPawn & (pieceAtt[us][knight] | pieceAtt[us][bishop])
for each victim in t: score += sgn * ThreatByMinor[victimType]   # e.g. p:5 n:57 b:57 r:77 q:52
# threat by our rook:
t = theirNonPawn & pieceAtt[us][rook]
for each victim: score += sgn * ThreatByRook[victimType]         # e.g. p:3 n:38 b:38 r:0 q:51
# safe pawn threat: our pawns (on safe squares) that attack a their piece:
safePawns = pawns[us] & (attackedBy[us] | ~attackedBy[them])
t = pawnAttacksFrom(safePawns) & own[them] & ~pawns[them]
score += sgn * 173/mg 94/eg * popcount(t)                        # ThreatBySafePawn, big
# pawn push threat: a pawn we can push to attack a their piece (push square safe):
score += sgn * ~48 * popcount(pushThreats)
# restricted: squares in enemy attack map we also attack but they defend (limits them):
restricted = attackedBy[them] & attackedBy[us] & ~... ; score += sgn * 7 * popcount(restricted)
# weak pieces (already have hanging): weakQueenProtection, king threat (attack on undefended piece by king)
```
Berserk/SF: this whole block is ~+30 Elo tuned. Start with ThreatByMinor + ThreatBySafePawn (the
two biggest), the rest after tuning.

### #5 Connected / phalanx / supported / backward pawns (into the pawn hash)
```
for each pawn p of side:
  phalanx  = has a friendly pawn on an adjacent file, same rank
  supported= number of friendly pawns defending p (0/1/2)
  if phalanx or supported:
     r = relative rank (2..7)
     bonus = Connected[r]   # SF11: {0,7,8,12,29,48,86} indexed by rank, ×
             (2 + bool(phalanx) + ... ) then + 21*supported ; big in mg
     mgP += sgn * bonus  (+ eg smaller)
  backward = cannot advance (stop sq attacked by enemy pawn) AND no friendly pawn behind on adj files
     if backward: mgP -= sgn * 9 ; egP -= sgn * 24 ; and if on open file extra rook pressure
```

### #7 Rook behind passed pawn / rook on 7th
```
for each rook r of side:
  # rook on relative 7th rank when enemy king on 8th or enemy pawns on 7th:
  if relRank(r)==7 and (relRank(enemyKing)>=7 or pawns[enemy] on their 2nd): mgRook += sgn*20; egRook += sgn*40
  # rook behind own OR enemy passed pawn on same file:
  for each passer pp:
     if fileOf(r)==fileOf(pp):
        behindOwn   = rook is on the side away from promotion  -> egPassed += sgn*  +tarraschBonus
        behindEnemy = rook attacks the enemy passer's path     -> egPassed += sgn*  smaller
```

## Depth-buying items (search) NOT already in our list, with evidence
- **Countermove heuristic** — 1 table [prevPiece][prevTo] -> move; adds a move-ordering slot
  between killers and history. Universally in strong engines, ~+5–15 Elo. Cheap. (We have cont-hist
  but not the explicit countermove ordering slot.) WORTH TRYING.
- **Capture history** (history table for captures indexed [piece][to][capturedType]) used to order
  captures beyond MVV/SEE and to gate qsearch/see pruning. Berserk/SF use it. ~+5–10. (We tried a
  variant in a bundle that came back neutral — retry standalone.)
- **Staged movegen** — generate TT move, then captures, then quiets lazily instead of full
  gen+sort every node. Pure speed (fewer wasted gens on cutoffs), no behavior change → more depth.
  Berserk/SF do this. ~+5–15 Elo as depth. Cost: med. Our tests of TT-shape were neutral, but this
  is orthogonal (it's about avoiding work, not changing the tree).
- **History-gated LMR/pruning** (reduce more when history score is very negative; SEE-gated quiet
  pruning `SEE < -margin*depth` in main search) — strong documented Elo in every engine. We have LMP
  but the SEE-gated quiet prune in the main tree is worth confirming present; if not, add it.

## The Berserk-specific "hardcore" note (user asked)
Berserk (pre-NNUE, v4–8) is not fundamentally different in *structure* — it's a bitboard HCE
alpha-beta engine like ours. What made it strong: (1) an unusually COMPLETE eval (every term above),
and (2) EVERYTHING tuned by its own gradient tuner on hundreds of millions of self-play + lichess
positions, hundreds of parameters, retuned every time a term was added. The lesson for us is #1 on
the ranked list: our ceiling is set less by which terms we have than by the fact that none are tuned.
Concretely: add the missing high-value terms (safe checks, scale factors, threats, connected pawns)
with reasonable hand weights, THEN build a tuner and let it set all weights from game data.
