cimport numpy as np
from cyperf.tools.types cimport bool, ITYPE_t, ITER
from cyperf.tools.getter cimport check_values

cpdef tuple reversed_index(ITER values)

cpdef bool is_strictly_increasing(ITER x) except? False

cpdef dict unique_index(ITER ll)


cdef class IndexedList:
    cdef:
        readonly list list
        readonly dict _index

    cpdef ITYPE_t index(self, value) except -1

    cpdef bool is_sorted(self)

    cpdef tuple sorted(self)

    cpdef IndexedList select(self, indices)

    cpdef tuple union(self, other, bool short_form=?)

    cpdef tuple intersection(self, other)

    cpdef tuple align(self, other)

    cpdef tuple difference(self, other)
