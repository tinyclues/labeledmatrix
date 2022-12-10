#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <functional>


template <typename T>
struct __attribute__ ((packed)) FLCouple {
    char* value;
    T index;
};

template <typename T>
std::function<bool(const FLCouple<T>, const FLCouple<T>)> make_str_greater_cmp(const size_t length){
    return [&length](const FLCouple<T> x, const FLCouple<T> y)->bool { return strncmp(x.value, y.value, length) > 0; };
};

template <typename T>
std::function<bool(const FLCouple<T>, const FLCouple<T>)> make_str_lesser_cmp(const size_t length){
    return [&length](const FLCouple<T> x, const FLCouple<T> y)->bool { return strncmp(x.value, y.value, length) < 0; };
};

