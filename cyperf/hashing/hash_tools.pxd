
cimport numpy as np
from cpython.string cimport PyString_AsString, PyString_Check
from cpython.object cimport PyObject
from libc.string cimport strncmp, memcpy


cdef extern from "farmhash.cc" nogil:
    unsigned int Hash32(const char* s, size_t len)
    unsigned int Hash32WithSeed(const char* s, size_t len, unsigned int seed)


cdef inline long python_string_length(char* ch, long size) nogil:
    cdef long i = size - 1

    while ch[i] == 0 and i >= 0:
        i -= 1

    return i + 1


cdef inline unsigned int composition_part(int residue, np.uint32_t* composition) nogil:
    """
    nb_group = len(composition) > 0
    total = sum(composition)
    """
    # cdef int residue = x % total
    cdef unsigned int i = 0
    residue -= composition[0]
    while residue >= 0:
        i += 1
        residue -= composition[i]
    return i


