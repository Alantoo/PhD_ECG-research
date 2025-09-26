from segment_artifacts_config import SegmentArtifactsConfig


class ArtifactsConfig:
    def __init__(self, cycles_count: int, segment_cfg: list[dict]):
        self.cycles_count = int(cycles_count)
        self.segment_cfg: list[SegmentArtifactsConfig] = list()
        for s in segment_cfg:
            self.segment_cfg.append(SegmentArtifactsConfig(s['points'], s['index'], s['count_or_pos'], s.get('exact_placement', False)))
