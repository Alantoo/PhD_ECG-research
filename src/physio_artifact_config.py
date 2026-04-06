class PhysioArtifactConfig:
    """A beat-level physiological artifact injected during modelling.

    Types:
        rhythm    — abnormal RR interval (rr_scale multiplies cycle duration)
        amplitude — abnormal QRS amplitude (amplitude_scale multiplies signal)
        shape     — abnormal morphology (noise_scale multiplies variance_scale)
    """
    def __init__(self, artifact_type, count_or_pos, exact_placement,
                 rr_scale=None, amplitude_scale=None, noise_scale=None):
        self.artifact_type   = str(artifact_type)
        self.count_or_pos    = int(count_or_pos)
        self.exact_placement = exact_placement is True
        self.rr_scale        = float(rr_scale)        if rr_scale        is not None else 1.8
        self.amplitude_scale = float(amplitude_scale) if amplitude_scale is not None else 0.3
        self.noise_scale     = float(noise_scale)     if noise_scale     is not None else 3.0
