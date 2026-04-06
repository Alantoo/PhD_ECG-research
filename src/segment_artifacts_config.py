class SegmentArtifactsConfig:
    def __init__(self, points, index, count_or_pos, exact_placement,
                 duration=None, min_height=None, max_height=None):
        self.points: list[list[int]] = points
        self.index = int(index)
        self.count_or_pos = int(count_or_pos)
        self.exact_placement = exact_placement is True
        # Optional render-time overrides
        self.duration   = float(duration) / 1000 if duration is not None else None  # UI sends ms, store as seconds
        self.min_height = float(min_height) if min_height is not None else None
        self.max_height = float(max_height) if max_height is not None else None
