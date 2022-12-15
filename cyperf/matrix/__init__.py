from cyperf.matrix.karma_sparse import np, is_karmasparse, linear_error_dense, DTYPE


def linear_error(matrix, regressor, intercept, target):
    def cast(x):
        return np.ascontiguousarray(x, dtype=DTYPE)

    if is_karmasparse(matrix):
        return matrix.linear_error(cast(regressor), cast(intercept), cast(target))
    else:
        return linear_error_dense(cast(matrix), cast(regressor), cast(intercept), cast(target))
