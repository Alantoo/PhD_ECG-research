import numpy as np


def _mad_z_scores(values):
    values = np.asarray(values, dtype=float)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    scale = 1.4826 * mad + 1e-9
    return np.abs(values - median) / scale


def detect_by_rhythm(r_peaks_sec, threshold_sigma=3.0):
    """Flag beats where the RR interval is an outlier (MAD-based z-score).

    Beat i is flagged when the interval between beat i-1 and beat i deviates
    from the median RR by more than threshold_sigma scaled MAD units.
    Beat 0 is never flagged by this detector alone.
    """
    r_peaks_sec = np.asarray(r_peaks_sec, dtype=float)
    if len(r_peaks_sec) < 2:
        return np.zeros(len(r_peaks_sec), dtype=bool)
    rr = np.diff(r_peaks_sec)
    z = _mad_z_scores(rr)
    mask = np.zeros(len(r_peaks_sec), dtype=bool)
    mask[1:] = z > threshold_sigma
    return mask


def detect_by_median_segment(segment_matrix, threshold_sigma=2.5, trim_ratio=0.3):
    """Flag beats whose segment shape deviates from a robust reference template.

    Builds the reference template from the most self-consistent beats (lowest
    RMSE against the initial median), so that a majority of anomalous beats
    cannot corrupt the template.

    segment_matrix: 2D array-like, shape (n_beats, n_samples).
    trim_ratio: fraction of most deviant beats excluded from the reference.
    """
    if len(segment_matrix) == 0:
        return np.array([], dtype=bool)
    min_len = min(len(row) for row in segment_matrix)
    matrix = np.array([row[:min_len] for row in segment_matrix], dtype=float)

    initial_template = np.median(matrix, axis=0)
    initial_rmse = np.sqrt(np.mean((matrix - initial_template) ** 2, axis=1))

    n_keep = max(1, int(len(matrix) * (1 - trim_ratio)))
    keep_idx = np.argsort(initial_rmse)[:n_keep]
    robust_template = np.median(matrix[keep_idx], axis=0)

    rmse = np.sqrt(np.mean((matrix - robust_template) ** 2, axis=1))
    return _mad_z_scores(rmse) > threshold_sigma


def detect_by_amplitude(r_amplitudes, threshold_sigma=3.0):
    """Flag beats whose R-peak amplitude is an outlier."""
    if len(r_amplitudes) == 0:
        return np.array([], dtype=bool)
    return _mad_z_scores(np.asarray(r_amplitudes, dtype=float)) > threshold_sigma


def detect_artifacts(r_peaks_sec, r_amplitudes, segment_matrices,
                     rhythm_sigma=3.0, segment_sigma=2.5, amplitude_sigma=3.0):
    """Run three detectors and return a combined result dict.

    Parameters
    ----------
    r_peaks_sec : array-like, shape (n_beats,)
        R-peak positions in seconds.
    r_amplitudes : array-like, shape (n_beats,)
        Signal amplitude at each R-peak sample.
    segment_matrices : list of array-like
        One matrix per segment type (P-R, R-T, T-P). Each row is one beat.
    rhythm_sigma, segment_sigma, amplitude_sigma : float
        Detection thresholds in MAD units.

    Returns
    -------
    dict with keys:
        artifact_beats         - union of all flagged beat indices
        rhythm_flags           - beat indices flagged by rhythm detector
        segment_flags          - beat indices flagged by segment detector
        amplitude_flags        - beat indices flagged by amplitude detector
        artifact_time_ranges   - [[start_sec, end_sec], ...] for overlay rendering
        total_beats            - total number of beats
        artifact_ratio         - fraction of beats flagged
    """
    n_beats = len(r_peaks_sec)
    r_peaks_arr = np.asarray(r_peaks_sec, dtype=float)

    rhythm_mask = detect_by_rhythm(r_peaks_arr, rhythm_sigma)

    seg_mask = np.zeros(n_beats, dtype=bool)
    for matrix in segment_matrices:
        if len(matrix) == 0:
            continue
        m = detect_by_median_segment(matrix, segment_sigma)
        length = min(len(m), n_beats)
        seg_mask[:length] |= m[:length]

    amp_mask = detect_by_amplitude(np.asarray(r_amplitudes, dtype=float), amplitude_sigma)
    if len(amp_mask) < n_beats:
        padded = np.zeros(n_beats, dtype=bool)
        padded[:len(amp_mask)] = amp_mask
        amp_mask = padded

    combined = rhythm_mask | seg_mask

    artifact_indices = np.where(combined)[0].tolist()
    rhythm_indices = np.where(rhythm_mask)[0].tolist()
    segment_indices = np.where(seg_mask)[0].tolist()
    amplitude_indices = np.where(amp_mask)[0].tolist()

    time_ranges = []
    for beat_idx in artifact_indices:
        start = float(r_peaks_arr[beat_idx - 1]) if beat_idx > 0 else 0.0
        end = float(r_peaks_arr[beat_idx + 1]) if beat_idx < n_beats - 1 else float(r_peaks_arr[-1])
        time_ranges.append([start, end])

    return {
        "artifact_beats": artifact_indices,
        "rhythm_flags": rhythm_indices,
        "segment_flags": segment_indices,
        "amplitude_flags": amplitude_indices,
        "artifact_time_ranges": time_ranges,
        "total_beats": n_beats,
        "artifact_ratio": round(len(artifact_indices) / n_beats, 4) if n_beats > 0 else 0.0,
    }
