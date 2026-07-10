"""Elo-difference estimate with a confidence interval from W/D/L game results,
instead of just reporting a raw win rate.

Uses the standard simple approach (as in cutechess-cli / common engine-testing
practice): treat each game's score (1/0.5/0) as one sample of a random
variable with mean p = score_sum/n, estimate the standard error of p directly
from the observed W/D/L counts (not the binomial p(1-p)/n approximation,
which is wrong in the presence of draws), then propagate that error through
the logistic Elo transform.
"""
import argparse
import math


def elo_diff(p):
    """Elo difference implied by a score fraction p in (0, 1)."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    return 400.0 * math.log10(p / (1.0 - p))


def estimate(wins, draws, losses, confidence=0.95):
    wins, draws, losses = int(round(wins)), int(round(draws)), int(round(losses))
    n = wins + draws + losses
    if n == 0:
        return None
    scores = [1.0] * wins + [0.5] * draws + [0.0] * losses
    p = sum(scores) / n
    # sample variance of the per-game score (correctly accounts for draws,
    # unlike a plain binomial p(1-p) which assumes only 0/1 outcomes)
    var = sum((s - p) ** 2 for s in scores) / n
    se_p = math.sqrt(var / n) if n > 1 else float("inf")

    e = elo_diff(p)
    # propagate the score's standard error through d(elo)/dp = 400/(ln10 * p*(1-p))
    p_clamped = min(max(p, 1e-6), 1 - 1e-6)
    d_elo_dp = 400.0 / (math.log(10) * p_clamped * (1.0 - p_clamped))
    se_elo = d_elo_dp * se_p

    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
    lo = elo_diff(max(p - z * se_p, 1e-6))
    hi = elo_diff(min(p + z * se_p, 1 - 1e-6))

    return {
        "n": n, "wins": wins, "draws": draws, "losses": losses,
        "score_pct": 100.0 * p,
        "elo_diff": e, "elo_lo": lo, "elo_hi": hi, "confidence": confidence,
    }


def format_estimate(label, est):
    if est is None:
        return f"{label}: no games"
    return (f"{label}: {est['wins']}W/{est['draws']}D/{est['losses']}L "
            f"(n={est['n']}, score={est['score_pct']:.1f}%) "
            f"Elo diff = {est['elo_diff']:+.0f} "
            f"[{est['elo_lo']:+.0f}, {est['elo_hi']:+.0f}] @ {int(est['confidence']*100)}% CI")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wins", type=float, required=True)
    p.add_argument("--draws", type=float, required=True)
    p.add_argument("--losses", type=float, required=True)
    p.add_argument("--label", default="result")
    p.add_argument("--confidence", type=float, default=0.95)
    args = p.parse_args()
    est = estimate(args.wins, args.draws, args.losses, args.confidence)
    print(format_estimate(args.label, est))


if __name__ == "__main__":
    main()
