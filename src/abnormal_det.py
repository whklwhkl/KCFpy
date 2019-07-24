class MovingAverage:
    def __init__(self, val, momentum=.9, conf_band=3):
        self.x = val
        self.x2 = val ** 2
        self.m = momentum
        self.m_ = 1 - momentum
        assert self.m > 0 and self.m_ > 0
        self.k = conf_band

    def __call__(self, val):
        std = max(1e-5, self.x2 - self.x ** 2) ** .5
        err = abs(val - self.x)
        self.x = self.m * self.x + self.m_ * val
        self.x2 = self.m * self.x2 + self.m_ * val ** 2
        return err > self.k * std   # True for abnormal
