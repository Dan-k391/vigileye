"""Statistics for the report. Everything resamples people, not clips, because
clips from one person are not independent of each other."""

import numpy as np


def auc(positives, negatives):
    """Chance a random positive outranks a random negative. Ties count half;
    0.5 means no better than guessing."""
    positives = np.asarray(positives, dtype=float)[:, None]
    negatives = np.asarray(negatives, dtype=float)[None, :]
    return float(np.mean((positives > negatives) + 0.5 * (positives == negatives)))


def paired_win_rate(pairs):
    """Share of people whose drowsy clip outscored their own alert clip."""
    wins = [1.0 if drowsy > alert else 0.5 if drowsy == alert else 0.0
            for drowsy, alert in pairs]
    return float(np.mean(wins))


def sensitivity_specificity(positive_flags, negative_flags):
    """Share of positives flagged, and share of negatives left unflagged."""
    return float(np.mean(positive_flags)), 1.0 - float(np.mean(negative_flags))


def bootstrap_ci(pairs, statistic, draws=2000, seed=0):
    """95% interval for a statistic of (drowsy value, alert value) per person."""
    pairs = list(pairs)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        picks = rng.integers(0, len(pairs), len(pairs))
        values.append(statistic([pairs[i] for i in picks]))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def pair_auc(pairs):
    drowsy, alert = zip(*pairs)
    return auc(drowsy, alert)


def pair_sensitivity(pairs):
    return float(np.mean([drowsy for drowsy, _ in pairs]))


def pair_specificity(pairs):
    return 1.0 - float(np.mean([alert for _, alert in pairs]))
