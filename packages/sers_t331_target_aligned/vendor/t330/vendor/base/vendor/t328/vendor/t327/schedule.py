"""One-based update schedule; independent of PyTorch and epoch rounding."""
import math


def learning_rate(update, total, warmup, maximum, minimum):
    if not (total >= 2 and 0 <= warmup < total and 1 <= update <= total):
        raise ValueError('Invalid schedule boundaries')
    if not 0 < minimum <= maximum:
        raise ValueError('Require 0 < minimum <= maximum')
    if warmup and update <= warmup:
        return maximum * update / warmup
    start = warmup if warmup else 1
    progress = (update - start) / (total - start)
    return minimum + (maximum - minimum) * (1 + math.cos(math.pi * progress)) / 2


class UpdateSchedule:
    def __init__(self, optimizer, total, warmup):
        self.optimizer, self.total, self.warmup = optimizer, total, warmup
        self.completed = 0
        # Validate immediately, rather than at the first optimizer update.
        for group in optimizer.param_groups:
            learning_rate(1, total, warmup, group['maximum_lr'], group['minimum_lr'])

    def prepare(self):
        for group in self.optimizer.param_groups:
            group['lr'] = learning_rate(self.completed + 1, self.total, self.warmup,
                                        group['maximum_lr'], group['minimum_lr'])

    def finish(self):
        self.completed += 1

    def state_dict(self):
        return dict(completed=self.completed, total=self.total, warmup=self.warmup)

    def load_state_dict(self, state):
        if (state['total'], state['warmup']) != (self.total, self.warmup):
            raise ValueError('Cannot change schedule when resuming')
        if not 0 <= state['completed'] <= self.total:
            raise ValueError('Invalid update count')
        self.completed = state['completed']
