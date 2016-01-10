# pylint: disable=invalid-name
"""learning rate scheduler"""

import math
import logging

class LearningRateScheduler(object):
    """Base class of learning rate scheduler"""
    def __init__(self):
        self.base_lr = 0.01

    def __call__(self, iteration):
        """
        Call to schedule current learning rate

        Parameters
        ----------
        iteration: int
            Current iteration count
        """
        raise NotImplementedError("must override this")


class ThreePhasedScheduler(LearningRateScheduler):
    """Reduce learning rate in factor

    Parameters
    ----------
    step: int
        schedule learning rate after every round
    factor: float
        reduce learning rate factor
    """
    def __init__(self, intermediateStartStep, intermediateEndStep, factor=10):
        super(ThreePhasedScheduler, self).__init__()
        if factor < 0.0:
            raise ValueError("Factor must be greater than 0.")
        self.intermediateStartStep = intermediateStartStep
        self.intermediateEndStep = intermediateEndStep
        self.factor = factor
        self.old_lr = self.base_lr
        self.init = False

    def __call__(self, iteration):
        """
        Call to schedule current learning rate

        Parameters
        ----------
        iteration: int
            Current iteration count
        """

        if self.init == False:
            self.init = True
            self.old_lr = self.base_lr

        if(iteration >= self.intermediateStartStep and iteration <= self.intermediateEndStep):
            lr = self.base_lr * self.factor;
        else:
            lr = self.base_lr;

        if lr != self.old_lr:
            self.old_lr = lr
            logging.info("At Iteration [%d]: Swith to new learning rate %.5f",
                         iteration, lr)
        return lr


class FactorScheduler(LearningRateScheduler):
    """Reduce learning rate in factor

    Parameters
    ----------
    step: int
        schedule learning rate after every round
    factor: float
        reduce learning rate factor
    """
    def __init__(self, step, factor=0.1):
        super(FactorScheduler, self).__init__()
        if step < 1:
            raise ValueError("Schedule step must be greater or equal than 1 round")
        if factor >= 1.0:
            raise ValueError("Factor must be less than 1 to make lr reduce")
        self.step = step
        self.factor = factor
        self.old_lr = self.base_lr
        self.init = False

    def __call__(self, iteration):
        """
        Call to schedule current learning rate

        Parameters
        ----------
        iteration: int
            Current iteration count
        """

        if self.init == False:
            self.init = True
            self.old_lr = self.base_lr
        lr = self.base_lr * math.pow(self.factor, int(iteration / self.step))
        if lr != self.old_lr:
            self.old_lr = lr
            logging.info("At Iteration [%d]: Swith to new learning rate %.5f",
                         iteration, lr)
        return lr


