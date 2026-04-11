class PhysioArtifactConfig:
    """A beat-level physiological artifact injected during modelling.

    Types:
        rhythm    — abnormal RR interval (rr_scale multiplies cycle duration)
        amplitude — abnormal QRS amplitude (amplitude_scale multiplies signal)
        shape     — abnormal morphology (noise_scale multiplies variance_scale)

    Zone targeting (optional):
        target_zones=None          — default behaviour (all zones for amplitude/shape;
                                     TP-only for rhythm in 6-zone mode)
        target_zones=[0, 2]        — apply only to zone indices 0 and 2
        target_zones='random'      — pick one zone per affected cycle (see random_zone_strategy)

        random_zone_strategy='re-roll'  — draw a new zone independently for each affected cycle
        random_zone_strategy='fixed'    — draw once; reuse the same zone for all affected cycles
    """
    def __init__(self, artifact_type, count_or_pos, exact_placement,
                 rr_scale=None, amplitude_scale=None, noise_scale=None,
                 target_zones=None, random_zone_strategy=None):
        self.artifact_type        = str(artifact_type)
        self.count_or_pos         = int(count_or_pos)
        self.exact_placement      = exact_placement is True
        self.rr_scale             = float(rr_scale)        if rr_scale        is not None else 1.8
        self.amplitude_scale      = float(amplitude_scale) if amplitude_scale is not None else 0.3
        self.noise_scale          = float(noise_scale)     if noise_scale     is not None else 3.0
        # None | list[int] | 'random'
        self.target_zones         = target_zones
        # 're-roll' | 'fixed'  (only meaningful when target_zones == 'random')
        self.random_zone_strategy = str(random_zone_strategy) if random_zone_strategy is not None else 're-roll'
