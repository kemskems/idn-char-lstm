class MeanAggregate:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0
        self.count = 0
        self.mean = 0

    def update(self, val, size=1):
        self.total += val * size
        self.count += size
        self.mean = self.total / self.count
