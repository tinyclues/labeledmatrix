from collections import Hashable
from functools import wraps

from numpy.random.mtrand import seed as np_seed
from numpy.random.mtrand import get_state as np_get_state
from numpy.random.mtrand import set_state as np_set_state
from random import seed as py_seed
from random import getstate as py_get_state
from random import setstate as py_set_state

import numpy as np


class use_seed:
    def __init__(self, seed=None):
        if seed is not None:
            if not isinstance(seed, int) and not isinstance(seed, np.int32):
                if isinstance(seed, Hashable):
                    self.seed = abs(np.int32(hash(seed)))  # hash returns int64, np.seed needs to be int32
                else:
                    raise ValueError("Invalid seed value `{}`, It should be an integer.".format(seed))
            elif seed < 0:
                raise ValueError("Invalid seed value `{}`, It should be positive.".format(seed))
            else:
                self.seed = seed
        else:
            self.seed = None

    def __enter__(self):
        self.np_state = np_get_state()
        self.py_state = py_get_state()
        if self.seed is not None:
            np_seed(self.seed)
            py_seed(self.seed)
        # Note: Returning self means that in "with ... as x", x will be self
        return self

    def __exit__(self, typ, val, _traceback):
        if self.seed is not None:
            np_set_state(self.np_state)
            py_set_state(self.py_state)

    def __call__(self, f):
        @wraps(f)
        def wrapper(*args, **kw):
            seed = self.seed if self.seed is not None else kw.pop('seed', None)
            with use_seed(seed):
                return f(*args, **kw)

        return wrapper
