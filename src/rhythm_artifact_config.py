import random

RHYTHM_CLASS_PREMATURE   = 'premature'
RHYTHM_CLASS_DELAYED     = 'delayed'
RHYTHM_CLASS_DROPPED     = 'dropped'
RHYTHM_CLASS_ALTERNATING = 'alternating'
RHYTHM_CLASS_IRREGULAR   = 'irregular'
RHYTHM_CLASS_CUSTOM      = 'custom'

SINGLE_CYCLE_CLASSES = {RHYTHM_CLASS_PREMATURE, RHYTHM_CLASS_DELAYED, RHYTHM_CLASS_DROPPED, RHYTHM_CLASS_CUSTOM}
WINDOW_CLASSES       = {RHYTHM_CLASS_ALTERNATING, RHYTHM_CLASS_IRREGULAR}
ALL_CLASSES          = SINGLE_CYCLE_CLASSES | WINDOW_CLASSES

DEFAULT_RR_RATIO = {
    RHYTHM_CLASS_PREMATURE: 0.75,
    RHYTHM_CLASS_DELAYED:   1.4,
}


class RhythmArtifactConfig:
    """A rhythm-structure artifact described relative to the mean cycle duration T0.

    Classes (see the thesis classification of rhythm artifacts):
        premature   — one cycle with R-R = rr_ratio * T0, rr_ratio < 1
        delayed     — one cycle with R-R = rr_ratio * T0, rr_ratio > 1
        dropped     — one cycle replaced by baseline so R-R ≈ 2 * T0
        alternating — `length` consecutive cycles alternating short_ratio / long_ratio
        irregular   — `length` consecutive cycles with rr_ratio drawn from [ratio_min, ratio_max],
                      each cycle dropped with probability drop_probability
        custom      — legacy behaviour: multiply the trailing TP zone by tp_scale

    Placement:
        single-cycle classes use count_or_pos / exact_placement (same as segment artifacts)
        window classes use start (1-based cycle, None = random) and length
    """

    def __init__(self, rhythm_class, count_or_pos=None, exact_placement=False,
                 rr_ratio=None, start=None, length=None,
                 short_ratio=None, long_ratio=None,
                 ratio_min=None, ratio_max=None, drop_probability=None,
                 tp_scale=None):
        rhythm_class = str(rhythm_class)
        if rhythm_class not in ALL_CLASSES:
            raise ValueError(f'unknown rhythm artifact class: {rhythm_class}')
        self.rhythm_class     = rhythm_class
        self.count_or_pos     = int(count_or_pos) if count_or_pos is not None else 1
        self.exact_placement  = exact_placement is True
        self.rr_ratio         = float(rr_ratio) if rr_ratio is not None else DEFAULT_RR_RATIO.get(rhythm_class, 1.0)
        self.start            = int(start) if start is not None else None
        self.length           = int(length) if length is not None else 6
        self.short_ratio      = float(short_ratio) if short_ratio is not None else 0.75
        self.long_ratio       = float(long_ratio) if long_ratio is not None else 1.3
        self.ratio_min        = float(ratio_min) if ratio_min is not None else 0.7
        self.ratio_max        = float(ratio_max) if ratio_max is not None else 1.5
        self.drop_probability = float(drop_probability) if drop_probability is not None else 0.0
        self.tp_scale         = float(tp_scale) if tp_scale is not None else 1.8

    def expand(self, cycles_count, pick_random_unique_n):
        """Return the per-cycle plan: list of dicts {cycle, class, rr_ratio, dropped, tp_scale}."""
        if self.rhythm_class in SINGLE_CYCLE_CLASSES:
            return self._expand_single(cycles_count, pick_random_unique_n)
        return self._expand_window(cycles_count)

    def _expand_single(self, cycles_count, pick_random_unique_n):
        if self.exact_placement:
            places = [self.count_or_pos - 1]
        else:
            places = pick_random_unique_n(cycles_count, self.count_or_pos)
        places = sorted(int(p) for p in places if 0 <= int(p) < cycles_count)

        plan = []
        for place in places:
            plan.append(self._entry(place))
        return plan

    def _expand_window(self, cycles_count):
        length = max(1, min(self.length, cycles_count))
        if self.start is not None:
            start = self.start - 1
        else:
            start = random.randint(0, max(0, cycles_count - length))
        start = max(0, min(start, cycles_count - 1))
        end   = min(cycles_count, start + length)

        plan = []
        for offset, cycle in enumerate(range(start, end)):
            if self.rhythm_class == RHYTHM_CLASS_ALTERNATING:
                ratio = self.short_ratio if offset % 2 == 0 else self.long_ratio
                plan.append(self._entry(cycle, rr_ratio=ratio))
            else:
                if self.drop_probability > 0 and random.random() < self.drop_probability:
                    plan.append(self._entry(cycle, dropped=True))
                else:
                    ratio = random.uniform(self.ratio_min, self.ratio_max)
                    plan.append(self._entry(cycle, rr_ratio=ratio))
        return plan

    def _entry(self, cycle, rr_ratio=None, dropped=None):
        if self.rhythm_class == RHYTHM_CLASS_DROPPED:
            dropped = True
        entry = {
            'cycle':    int(cycle),
            'class':    self.rhythm_class,
            'dropped':  bool(dropped),
            'rr_ratio': None,
            'tp_scale': None,
        }
        if dropped:
            return entry
        if self.rhythm_class == RHYTHM_CLASS_CUSTOM:
            entry['tp_scale'] = self.tp_scale
        else:
            entry['rr_ratio'] = float(rr_ratio if rr_ratio is not None else self.rr_ratio)
        return entry
