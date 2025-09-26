class SegmentArtifactsConfig:
    def __init__(self, points, index, count_or_pos, exact_placement):
        self.points: list[list[int]] = points
        self.index = int(index)
        self.count_or_pos = int(count_or_pos)
        self.exact_placement = exact_placement is True
