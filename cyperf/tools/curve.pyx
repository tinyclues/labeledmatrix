# cython: embedsignature=True
# cython: nonecheck=True
# cython: overflowcheck=True
# cython: boundscheck=False
# cython: wraparound=False

cimport numpy as np
import numpy as np
from libc.math cimport round as cround


__all__ = ['cy_compute_serie']


def cy_compute_serie(x, y, precision=2, concave=False):
    assert x.shape[0] == y.shape[0]
    x_r, y_r = round_up(np.asarray(x, dtype=np.float64),
                        np.asarray(y, dtype=np.float64),
                        precision)

    if concave:
        length = len(x_r) + 1
        while length > len(x_r):
            length = len(x_r)
            x_r, y_r = make_more_concave(x_r, y_r)
    return x_r, y_r


cdef inline double my_round(const double a, const double scale) nogil:
    return cround(a * scale) / scale


def round_up(double[:] x, double[:] y, int precision=2):
    assert x.shape[0] == y.shape[0]

    cdef double scale = 10. ** precision, scale_up = 10. ** (precision + 1)
    cdef long i, j = 0, nb = x.shape[0]
    cdef double[::1] xx = np.zeros(nb, dtype=np.float64), yy = np.zeros(nb, dtype=np.float64)
    cdef double old_xxi = -1, old_yyi = -1, xxi, yyi, displayed_xxi, displayed_yyi

    with nogil:
        for i in xrange(nb):
            xxi = my_round(x[i], scale)
            yyi = my_round(y[i], scale)
            if xxi != old_xxi or yyi != old_yyi:
                displayed_xxi = my_round(x[i], scale_up)
                displayed_yyi = my_round(y[i], scale_up)
                if j == 0 or xx[j - 1] != displayed_xxi or yy[j - 1] != displayed_yyi:
                    xx[j] = displayed_xxi
                    yy[j] = displayed_yyi
                    j += 1
                old_xxi, old_yyi = xxi, yyi

    return np.asarray(xx)[:j], np.asarray(yy)[:j]


def make_more_concave(double[:] x, double[:] y):
    assert x.shape[0] == y.shape[0]
    cdef:
        np.uint8_t[:] mask = np.ones_like(x, dtype=np.uint8)
        long i, n = x.shape[0]

    with nogil:
        for i in xrange(1, n - 1):
            if (y[i] - y[i-1]) * (x[i+1] - x[i-1]) <= \
               (x[i] - x[i-1]) * (y[i+1] - y[i-1]):
                mask[i] = 0

    return (np.asarray(x)[np.asarray(mask, dtype=np.bool)], np.asarray(y)[np.asarray(mask, dtype=np.bool)])
