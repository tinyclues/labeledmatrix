# cython: embedsignature=True
# cython: nonecheck=True
# cython: overflowcheck=True
# cython: unraisable_tracebacks=True
# cython: infer_types=True

import numpy as np
from cyperf.tools.types import ITYPE

include "indices_where.pxi"
