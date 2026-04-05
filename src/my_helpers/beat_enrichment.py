import numpy as np
from scipy.interpolate import interp1d

# Recommended MAD-sigma thresholds per delineation method (tuned for typical noise levels).
METHOD_SIGMA_DEFAULTS = {
    'dwt':    2.5,
    'cwt':    2.8,
    'peak':   3.0,
    'merged': 2.5,
}


def _amp_at(sig, t_sec, sampling_rate, time_offset):
    """Signal amplitude at a given absolute time in seconds."""
    if t_sec is None:
        return None
    idx = int(round((float(t_sec) - time_offset) * sampling_rate))
    idx = max(0, min(idx, len(sig) - 1))
    return round(float(sig[idx]), 4)


def _midpoint(a, b):
    if a is not None and b is not None:
        return (float(a) + float(b)) / 2.0
    return None


def compute_waveform(beat, prev_beat, sig_start, sig_end):
    start = beat.get('p_on') or beat.get('q') or beat.get('r')
    if start is None and prev_beat is not None:
        start = (prev_beat.get('t_off') or prev_beat.get('t_on') or
                 prev_beat.get('s') or prev_beat.get('r'))
    if start is None:
        start = sig_start

    end = beat.get('t_off') or beat.get('t_on') or beat.get('s') or beat.get('r')
    if end is None:
        end = sig_end

    return {
        'start': round(float(start), 4) if start is not None else None,
        'end':   round(float(end), 4)   if end is not None else None,
    }


def compute_tp(beat, beats, beat_idx, sig_end):
    """TP segment: t_off[i] → earliest fiducial of the next beat in the array."""
    t_off = beat.get('t_off')
    if t_off is None:
        return None

    next_start = None
    for j in range(beat_idx + 1, len(beats)):
        nb = beats[j]
        cand = nb.get('p_on') or nb.get('q') or nb.get('r')
        if cand is not None:
            next_start = cand
            break

    if next_start is None:
        next_start = sig_end

    if next_start is None or float(next_start) <= float(t_off):
        return None
    return {'start': round(float(t_off), 4), 'end': round(float(next_start), 4)}


def compute_amplitudes(beat, sig, sampling_rate, time_offset):
    at = lambda t: _amp_at(sig, t, sampling_rate, time_offset)
    return {
        'p': at(_midpoint(beat.get('p_on'), beat.get('p_off'))),
        'q': at(beat.get('q')),
        'r': at(beat.get('r')),
        's': at(beat.get('s')),
        't': at(_midpoint(beat.get('t_on'), beat.get('t_off'))),
    }


def compute_intervals(beat, prev_beat):
    r      = beat.get('r')
    p_on   = beat.get('p_on')
    q      = beat.get('q')
    s      = beat.get('s')
    t_off  = beat.get('t_off')
    prev_r = prev_beat.get('r') if prev_beat is not None else None

    rr  = round(float(r) - float(prev_r), 4) if r is not None and prev_r is not None else None
    pr  = round(float(r) - float(p_on), 4)   if r is not None and p_on is not None else None
    qrs = round(float(s) - float(q), 4)       if s is not None and q is not None else None
    qt  = round(float(t_off) - float(q), 4)   if t_off is not None and q is not None else None
    qtc = (round(float(qt) / np.sqrt(float(rr)), 4)
           if qt is not None and rr is not None and float(rr) > 0 else None)

    return {'rr': rr, 'pr': pr, 'qrs': qrs, 'qt': qt, 'qtc': qtc}


def compute_baseline(beat, prev_beat, sig, sampling_rate, time_offset):
    """Estimate isoelectric baseline from the TP segment preceding this beat."""
    p_on       = beat.get('p_on')
    prev_t_off = prev_beat.get('t_off') if prev_beat is not None else None

    if prev_t_off is not None and p_on is not None and float(p_on) > float(prev_t_off):
        s = int(round((float(prev_t_off) - time_offset) * sampling_rate))
        e = int(round((float(p_on)       - time_offset) * sampling_rate))
        s, e = max(0, s), max(0, e)
        seg = sig[s:e]
        if len(seg) >= 2:
            return round(float(np.median(seg)), 4)

    # Fallback: 20 ms window immediately before p_on
    if p_on is not None:
        e = int(round((float(p_on) - time_offset) * sampling_rate))
        s = max(0, e - max(1, int(0.02 * sampling_rate)))
        seg = sig[s:e]
        if len(seg) >= 1:
            return round(float(np.median(seg)), 4)

    return None


def compute_st_deviation(beat, sig, sampling_rate, time_offset, baseline):
    s_val = beat.get('s')
    t_on  = beat.get('t_on')
    if s_val is None or t_on is None or baseline is None:
        return None
    si = int(round((float(s_val) - time_offset) * sampling_rate))
    ti = int(round((float(t_on)  - time_offset) * sampling_rate))
    si, ti = max(0, si), max(0, ti)
    seg = sig[si:ti]
    if len(seg) < 2:
        return None
    return round(float(np.mean(seg)) - float(baseline), 4)


def compute_morphology(beat, amplitudes, intervals, baseline):
    r_amp  = amplitudes.get('r')
    t_amp  = amplitudes.get('t')
    qrs    = intervals.get('qrs')

    inverted_t = bool(
        r_amp is not None and t_amp is not None and baseline is not None and
        float(t_amp) < float(baseline) and float(r_amp) > float(baseline)
    )
    wide_qrs  = bool(qrs is not None and float(qrs) > 0.120)
    missing_p = beat.get('p_on') is None or beat.get('p_off') is None

    return {'inverted_t': inverted_t, 'wide_qrs': wide_qrs, 'missing_p': missing_p}


def compute_template_correlations(beats, sig, sampling_rate, time_offset, fixed_len=200):
    """Pearson correlation of each beat's full waveform (p_on→t_off) vs the median template."""
    valid_idxs = []
    resampled  = []

    for i, b in enumerate(beats):
        p_on  = b.get('p_on')
        t_off = b.get('t_off')
        if p_on is None or t_off is None or float(t_off) <= float(p_on):
            continue
        s   = max(0, int(round((float(p_on)  - time_offset) * sampling_rate)))
        e   = max(0, int(round((float(t_off) - time_offset) * sampling_rate)))
        seg = sig[s:e]
        if len(seg) < 4:
            continue
        f = interp1d(np.linspace(0.0, 1.0, len(seg)), seg, kind='linear')
        valid_idxs.append(i)
        resampled.append(f(np.linspace(0.0, 1.0, fixed_len)))

    if len(resampled) < 2:
        return {}

    mat      = np.array(resampled, dtype=float)
    template = np.median(mat, axis=0)
    t_std    = float(np.std(template))

    result = {}
    for i, seg in zip(valid_idxs, resampled):
        s_std = float(np.std(seg))
        if s_std < 1e-10 or t_std < 1e-10:
            corr = 1.0
        else:
            corr = float(np.corrcoef(seg, template)[0, 1])
        result[i] = round(float(np.clip(corr, 0.0, 1.0)), 3)

    return result


def compute_quality(beat, template_correlation):
    if template_correlation is None:
        return None
    total_zones      = 6  # p_on, p_off, q, s, t_on, t_off
    missing          = beat.get('missing_zones') or []
    missing_fraction = len(missing) / total_zones
    q = float(template_correlation) * (1.0 - missing_fraction * 0.5)
    return round(float(np.clip(q, 0.0, 1.0)), 3)


def compute_delta(beat, prev_beat):
    """Beat-to-beat deltas: change in RR interval and R-peak amplitude vs the previous beat."""
    rr      = (beat.get('intervals') or {}).get('rr')
    prev_rr = ((prev_beat.get('intervals') or {}).get('rr')) if prev_beat else None
    amp_r      = (beat.get('amplitudes') or {}).get('r')
    prev_amp_r = ((prev_beat.get('amplitudes') or {}).get('r')) if prev_beat else None
    return {
        'rr':          round(float(rr) - float(prev_rr), 4) if rr is not None and prev_rr is not None else None,
        'amplitude_r': round(float(amp_r) - float(prev_amp_r), 4) if amp_r is not None and prev_amp_r is not None else None,
    }


def compute_consensus(waves_dwt, waves_cwt, sampling_rate, time_offset, n_beats, tolerance_ms=20):
    """Compare DWT vs CWT per-zone fiducial detection.

    Both waves_dwt / waves_cwt are raw dicts from nk.ecg_delineate —
    keys like 'ECG_P_Peaks', each a list of sample indices (may contain NaN).

    Returns a consensus summary dict with per-zone mean absolute differences
    and a list of zones where the two methods disagree by more than tolerance_ms.
    """
    tolerance = tolerance_ms / 1000.0
    zone_keys = [
        'ECG_P_Onsets', 'ECG_P_Offsets',
        'ECG_T_Onsets', 'ECG_T_Offsets',
        'ECG_Q_Peaks', 'ECG_S_Peaks',
        'ECG_P_Peaks', 'ECG_T_Peaks',
    ]

    def to_sec_arr(waves, key):
        raw = waves.get(key, [])
        arr = np.array(raw, dtype=float)
        if len(arr) < n_beats:
            arr = np.concatenate([arr, np.full(n_beats - len(arr), np.nan)])
        elif len(arr) > n_beats:
            arr = arr[:n_beats]
        return np.where(np.isnan(arr), np.nan, arr / sampling_rate + time_offset)

    zone_mean_diffs = {}
    uncertain_zones = []
    n_agreed = 0
    n_total  = 0

    for zone in zone_keys:
        a = to_sec_arr(waves_dwt, zone)
        b = to_sec_arr(waves_cwt, zone)
        both_valid = ~np.isnan(a) & ~np.isnan(b)
        if not np.any(both_valid):
            continue
        diffs     = np.abs(a[both_valid] - b[both_valid])
        mean_diff = float(np.mean(diffs))
        zone_mean_diffs[zone] = round(mean_diff, 5)
        n_total += 1
        if mean_diff < tolerance:
            n_agreed += 1
        else:
            uncertain_zones.append(zone)

    agreement_rate = round(n_agreed / n_total, 3) if n_total > 0 else None
    return {
        'method_a':            'dwt',
        'method_b':            'cwt',
        'agreement_rate':      agreement_rate,
        'uncertain_zones':     uncertain_zones,
        'zone_mean_diffs_sec': zone_mean_diffs,
    }


def compute_gaps(beats, sig_start, sig_end):
    gaps = []
    i    = 0
    n    = len(beats)
    while i < n:
        if not beats[i].get('skipped'):
            i += 1
            continue

        group_start = i
        while i < n and beats[i].get('skipped'):
            i += 1
        group_end = i - 1

        prev_b = beats[group_start - 1] if group_start > 0 else None
        next_b = beats[i]               if i < n else None

        gap_start = None
        if prev_b:
            bd = prev_b.get('waveform') or {}
            gap_start = prev_b.get('t_off') or prev_b.get('t_on') or prev_b.get('s') or bd.get('end')
        if gap_start is None:
            gap_start = sig_start

        gap_end = None
        if next_b:
            bd = next_b.get('waveform') or {}
            gap_end = next_b.get('p_on') or next_b.get('q') or next_b.get('r') or bd.get('start')
        if gap_end is None:
            gap_end = sig_end

        indices   = list(range(group_start, group_end + 1))
        n_missing = len(indices)
        prev_r    = prev_b.get('r') if prev_b else None
        next_r    = next_b.get('r') if next_b else None

        r_estimated = []
        if prev_r is not None and next_r is not None and n_missing > 0:
            step        = (float(next_r) - float(prev_r)) / (n_missing + 1)
            r_estimated = [round(float(prev_r) + step * (k + 1), 4) for k in range(n_missing)]

        gaps.append({
            'start':             round(float(gap_start), 4) if gap_start is not None else None,
            'end':               round(float(gap_end),   4) if gap_end   is not None else None,
            'beat_indices':      indices,
            'r_peaks_estimated': r_estimated if r_estimated else None,
        })

    return gaps


def compute_zone_stats(beats):
    non_skipped = [b for b in beats if not b.get('skipped')]
    n = len(non_skipped)
    if n == 0:
        return {}

    def zone_stat(f_on, f_off):
        detected  = [b for b in non_skipped if b.get(f_on) is not None and b.get(f_off) is not None]
        durations = [float(b[f_off]) - float(b[f_on]) for b in detected]
        return {
            'detection_rate': round(len(detected) / n, 3),
            'mean_duration':  round(float(np.mean(durations)), 4) if durations else None,
        }

    def tp_stat():
        detected  = [b for b in non_skipped if b.get('tp') is not None]
        durations = [float(b['tp']['end']) - float(b['tp']['start']) for b in detected]
        return {
            'detection_rate': round(len(detected) / n, 3),
            'mean_duration':  round(float(np.mean(durations)), 4) if durations else None,
        }

    return {
        'p_wave':     zone_stat('p_on', 'p_off'),
        'qrs':        zone_stat('q', 's'),
        't_wave':     zone_stat('t_on', 't_off'),
        'tp_segment': tp_stat(),
    }


def compute_hrv(beats):
    rr_list = []
    for b in beats:
        ivs = b.get('intervals')
        if ivs:
            rr = ivs.get('rr')
            if rr is not None:
                rr_list.append(float(rr))

    if len(rr_list) < 2:
        return None

    rr    = np.array(rr_list, dtype=float)
    diffs = np.diff(rr)
    nn50  = int(np.sum(np.abs(diffs) > 0.050))

    return {
        'mean_rr': round(float(np.mean(rr)), 4),
        'sdnn':    round(float(np.std(rr, ddof=1)), 4),
        'rmssd':   round(float(np.sqrt(np.mean(diffs ** 2))), 4),
        'pnn50':   round(nn50 / len(rr), 3),
    }


def enrich_beats(beats_raw, sig_arr, sampling_rate, sig_start, sig_end, time_offset=0.0):
    """Enrich beat dicts in-place with computed analytical fields.

    Pass 1 — per-beat (no cross-beat aggregation needed):
        waveform, tp, status, amplitudes, intervals,
        baseline, st_deviation, morphology.

    Pass 2 — requires all beats (template correlation + quality).
    """
    for i, b in enumerate(beats_raw):
        prev_b = beats_raw[i - 1] if i > 0 else None

        b['waveform'] = compute_waveform(b, prev_b, sig_start, sig_end)
        b['tp']         = compute_tp(b, beats_raw, i, sig_end)

        missing = b.get('missing_zones') or []
        if b.get('skipped'):
            b['status'] = 'skipped'
        elif missing:
            b['status'] = 'partial'
        else:
            b['status'] = 'complete'

        b['amplitudes']   = compute_amplitudes(b, sig_arr, sampling_rate, time_offset)
        b['intervals']    = compute_intervals(b, prev_b)
        baseline          = compute_baseline(b, prev_b, sig_arr, sampling_rate, time_offset)
        b['baseline']     = baseline
        b['st_deviation'] = compute_st_deviation(b, sig_arr, sampling_rate, time_offset, baseline)
        b['morphology']   = compute_morphology(b, b['amplitudes'], b['intervals'], baseline)
        b['delta']        = compute_delta(b, prev_b)

    tc_map = compute_template_correlations(beats_raw, sig_arr, sampling_rate, time_offset)
    for i, b in enumerate(beats_raw):
        tc              = tc_map.get(i)
        b['template_correlation'] = tc
        b['quality']              = compute_quality(b, tc)

    return beats_raw
