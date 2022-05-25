
"""
Some utility functions for deep-measure.
"""

from datetime import datetime
import numpy as np


class EarlyStop:
    """
    A simple early stopping checker.
    Log the loss (the lower the better) into the buffer with length of 2 * window_size.
    If the AVG of older half of the buffer is lower than the AVG of newer half of the buffer, return a False signal.
    When called, return current keep-going flag.

    True: keep going
    False: terminate

    """
    def __init__(self, name='', window_size=5):

        self.name = 'name'
        self.window_size = window_size
        self.buffer = []
        self.FLAG_KEEP_GOING = True

    def update(self, loss_new):
        """
        Update the early-stop buffer.

        If this early-stop object's flag has been set to False, then the buffer will not be updated.

        :param loss_new: float. The new loss to be logged into the buffer
        :return: return_signal:
                    True: keep going;
                    False: stop
        """
        if not self.FLAG_KEEP_GOING:
            return False
        if len(self.buffer) < (self.window_size * 2):
            self.buffer.append(loss_new)
            return_signal = True
        else:
            self.buffer.pop(0)
            self.buffer.append(loss_new)
            if np.average(self.buffer[:self.window_size]) <= np.average(self.buffer[self.window_size:]):
                self.buffer = []
                return_signal = False
            else:
                return_signal = True
        if not return_signal:
            print(f'\n{datetime.now()} I === Early stop triggered. === name: {self.name}Last value: {loss_new}')
        self.FLAG_KEEP_GOING = return_signal
        return return_signal

    def set_terminate(self):
        self.FLAG_KEEP_GOING = False

    def set_keep_going(self):
        self.FLAG_KEEP_GOING = True

    def __call__(self):
        return self.FLAG_KEEP_GOING
