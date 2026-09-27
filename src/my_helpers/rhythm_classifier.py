"""Rhythm-structure artifact classifier over the R-R interval sequence.

Classes follow the thesis taxonomy of rhythm artifacts:
    0 normal
    1 premature   — R-R < (1 - delta) * T0
    2 delayed     — R-R > (1 + delta) * T0
    3 dropped     — R-R ≈ k * T0, k >= 2 (one or more missing cycles)
    4 alternating — run of >= min_alternating_run intervals alternating short / long
    5 irregular   — window where >= irregular_fraction of intervals are abnormal without alternation

T0 is a local reference: the median of the neighbouring intervals (excluding the
interval itself) after one outlier-removal pass, so heart-rate drift and dense
artifact windows do not mask each other the way a global median + MAD does.

Interval i runs from R_i to R_{i+1} and is attributed to beat i (the beat whose
following pause is abnormal), matching how the modelling module places rhythm artifacts.
"""

import numpy as np

CLASS_NORMAL      = 0
CLASS_PREMATURE   = 1
CLASS_DELAYED     = 2
CLASS_DROPPED     = 3
CLASS_ALTERNATING = 4
CLASS_IRREGULAR   = 5

CLASS_NAMES = {
    CLASS_NORMAL:      'normal',
    CLASS_PREMATURE:   'premature',
    CLASS_DELAYED:     'delayed',
    CLASS_DROPPED:     'dropped',
    CLASS_ALTERNATING: 'alternating',
    CLASS_IRREGULAR:   'irregular',
}

DEFAULT_DELTA               = 0.2
DEFAULT_DROPPED_EPS         = 0.2
DEFAULT_WINDOW              = 12
DEFAULT_MIN_ABS_DEVIATION   = 0.04   # seconds — ignore sampling jitter on very short T0
DEFAULT_MIN_ALTERNATING_RUN = 4
DEFAULT_IRREGULAR_WINDOW    = 8
DEFAULT_IRREGULAR_FRACTION  = 0.5
DEFAULT_ALTERNATING_TOLERANCE = 0.12   # max spread of short (and of long) ratios inside an alternating run


def classify_rr_sequence(r_times, delta=DEFAULT_DELTA, dropped_eps=DEFAULT_DROPPED_EPS,
                         window=DEFAULT_WINDOW, min_abs_deviation=DEFAULT_MIN_ABS_DEVIATION,
                         min_alternating_run=DEFAULT_MIN_ALTERNATING_RUN,
                         irregular_window=DEFAULT_IRREGULAR_WINDOW,
                         irregular_fraction=DEFAULT_IRREGULAR_FRACTION):
    """Classify every R-R interval of a beat sequence.

    r_times: R-peak times in seconds, ascending.
    Returns dict with:
        intervals: list of {beat_idx, rr, t0, ratio, deviation, cls, dropped_cycles}
        classes:   per-interval class list (len = len(r_times) - 1)
        stats:     parameters and global summary
    """
    r = np.asarray(r_times, dtype=float)
    if len(r) < 3:
        return {'intervals': [], 'classes': [], 'stats': _stats(delta, dropped_eps, window, [], 0.0)}

    rr = np.diff(r)
    n  = len(rr)
    t0 = np.array([_local_reference(rr, i, window, delta) for i in range(n)])
    ratio = rr / np.where(t0 > 0, t0, 1e-9)
    deviation = np.abs(rr - t0)

    classes = np.full(n, CLASS_NORMAL, dtype=int)
    dropped_cycles = np.zeros(n, dtype=int)
    for i in range(n):
        if deviation[i] < min_abs_deviation:
            continue
        if ratio[i] >= 2.0 - dropped_eps:
            classes[i] = CLASS_DROPPED
            dropped_cycles[i] = max(1, int(round(ratio[i])) - 1)
        elif ratio[i] > 1.0 + delta:
            classes[i] = CLASS_DELAYED
        elif ratio[i] < 1.0 - delta:
            classes[i] = CLASS_PREMATURE

    _mark_alternating(classes, ratio, min_alternating_run)
    _mark_irregular(classes, irregular_window, irregular_fraction)

    intervals = []
    for i in range(n):
        intervals.append({
            'beat_idx':       int(i),
            'rr':             round(float(rr[i]), 4),
            't0':             round(float(t0[i]), 4),
            'ratio':          round(float(ratio[i]), 3),
            'deviation':      round(float(deviation[i]), 4),
            'cls':            int(classes[i]),
            'class_name':     CLASS_NAMES[int(classes[i])],
            'dropped_cycles': int(dropped_cycles[i]),
        })

    return {
        'intervals': intervals,
        'classes':   classes.tolist(),
        'stats':     _stats(delta, dropped_eps, window, classes, float(np.median(rr))),
    }


def _local_reference(rr, i, window, delta):
    """Mean cycle duration T0 estimated from the neighbours of interval i.

    T0 is the *mean* cycle (as in the thesis definition), so an alternating or
    irregular window still yields the true cycle length instead of collapsing onto
    the short or the long cluster. Multi-cycle gaps (dropped beats) are divided by
    the number of cycles they span before averaging, and the remaining values are
    winsorised to ±50 % of the preliminary median so a single extreme interval
    cannot drag the estimate.
    """
    half = max(1, window // 2)
    lo, hi = max(0, i - half), min(len(rr), i + half + 1)
    neighbours = np.concatenate([rr[lo:i], rr[i + 1:hi]])
    if len(neighbours) == 0:
        return float(rr[i])
    preliminary = float(np.median(neighbours))
    if preliminary <= 0:
        return float(rr[i])
    spans = np.maximum(1, np.round(neighbours / preliminary))
    spans = np.where(neighbours / preliminary >= 2.0 - delta, spans, 1)
    per_cycle = neighbours / spans
    clipped = np.clip(per_cycle, 0.5 * preliminary, 1.5 * preliminary)
    return float(np.mean(clipped))


def _mark_alternating(classes, ratio, min_run, tolerance=DEFAULT_ALTERNATING_TOLERANCE):
    """Relabel sign-alternating runs of short / long intervals as alternating.

    A run counts only when its short intervals resemble each other and its long
    intervals resemble each other (within `tolerance` of the run's own means);
    otherwise the run is a chaotic mix and is left for the irregular rule.
    """
    n = len(classes)
    i = 0
    while i < n:
        if classes[i] not in (CLASS_PREMATURE, CLASS_DELAYED):
            i += 1
            continue
        j = i
        while j + 1 < n and classes[j + 1] in (CLASS_PREMATURE, CLASS_DELAYED) \
                and np.sign(ratio[j + 1] - 1.0) == -np.sign(ratio[j] - 1.0):
            j += 1
        run = j - i + 1
        if run >= min_run and _run_is_regular(ratio[i:j + 1], tolerance):
            classes[i:j + 1] = CLASS_ALTERNATING
        i = j + 1


def _run_is_regular(run_ratios, tolerance):
    shorts = run_ratios[run_ratios < 1.0]
    longs  = run_ratios[run_ratios > 1.0]
    for group in (shorts, longs):
        if len(group) and np.max(np.abs(group - np.mean(group))) > tolerance:
            return False
    return True


def _mark_irregular(classes, window, fraction):
    n = len(classes)
    if n < window:
        return
    abnormal = np.isin(classes, [CLASS_PREMATURE, CLASS_DELAYED, CLASS_DROPPED]).astype(int)
    irregular = np.zeros(n, dtype=bool)
    for start in range(0, n - window + 1):
        seg = slice(start, start + window)
        if abnormal[seg].sum() >= fraction * window:
            irregular[seg] = True
    for i in range(n):
        if irregular[i] and abnormal[i]:
            classes[i] = CLASS_IRREGULAR


def _stats(delta, dropped_eps, window, classes, median_rr):
    classes = np.asarray(classes, dtype=int)
    counts = {CLASS_NAMES[c]: int((classes == c).sum()) for c in CLASS_NAMES if c != CLASS_NORMAL}
    return {
        'delta':        delta,
        'dropped_eps':  dropped_eps,
        'window':       window,
        'median_rr':    round(median_rr, 4),
        'n_intervals':  int(len(classes)),
        'class_counts': counts,
    }
