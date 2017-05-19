
cimport numpy as np
from cpython.string cimport PyString_AsString, PyString_Check
from cpython.object cimport PyObject
from libc.string cimport strncmp, memcpy


cdef extern from "farmhash.cc" nogil:
    inline signed int Hash32(const char* s, size_t len)
    inline signed int Hash32WithSeed(const char* s, size_t len, signed int seed)


cdef inline signed int composition_part(int residue, np.uint32_t* composition) nogil:
    """
    nb_group = len(composition) > 0
    total = sum(composition)
    """
    # cdef int residue = x % total
    cdef signed int i = 0
    residue -= composition[0]
    while residue >= 0:
        i += 1
        residue -= composition[i]
    return i


