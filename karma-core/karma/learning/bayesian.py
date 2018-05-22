import numpy as np

from karma.core.utils import create_timer
from karma.learning.utils import BasicVirtualHStack
from karma.learning.matrix_utils import diagonal_of_inverse_symposdef
from karma.learning.regression import linear_regression_coefficients


def linear_coefficients_and_posteriori(X, y, w_priori=None, intercept_priori=0.,
                                       C_priori=1e5, intercept_C_priori=1e10,
                                       sample_weight=None, full_hessian=True,
                                       noise_variance=1., timer=None):
    if timer is None:
        timer = create_timer(None)

    conv_dict = {}

    with timer('BayLinReg_Reg_Init'):
        if not isinstance(X, BasicVirtualHStack):
            X = BasicVirtualHStack(X)
        XX = X.materialize()
        XX = np.hstack((XX, np.ones((XX.shape[0], 1), dtype=np.float32)))

        w_priori = np.hstack((X.adjust_array_to_total_dimension(w_priori, 'w_priori'), intercept_priori))
        C_priori = np.hstack((X.adjust_array_to_total_dimension(C_priori, 'C_priori'), intercept_C_priori))

        noise_precision = 1. / noise_variance  # beta in Bishop's Pattern Recogn and ML (3.3.1)

        # remove from y the part coming from the prior
        y_hat_prior = XX.dot(w_priori)
        y_prime = y - y_hat_prior

        # rescale and center x
        X_prime = XX * np.sqrt(C_priori)

    with timer('BayLinReg_Reg_Mean'):
        # get the MLE (==mean since gaussian)
        y_hat_post_prime, _, w_post_prime = linear_regression_coefficients(X_prime, y_prime,
                                                                           intercept=False,
                                                                           C=noise_precision,
                                                                           sample_weight=sample_weight)
        w_post = w_post_prime * np.sqrt(C_priori) + w_priori
        intercept_post = w_post[-1]
        w_post = X.split_by_dims(w_post[:-1])

    with timer('BayLinReg_Reg_Variance'):
        # get the variance
        if full_hessian:
            inv_cov = np.diag(1. / C_priori)
            inv_cov += XX.transpose().dot(XX) * noise_precision
            cov_post_diag = diagonal_of_inverse_symposdef(inv_cov)
        else:
            cov_post_diag = 1. / (1. / C_priori + np.einsum('ij,ij->j', XX, XX) * noise_precision)
        intercept_C_post = cov_post_diag[-1]
        w_C_post = X.split_by_dims(cov_post_diag[:-1])

    y_hat_post = y_hat_post_prime + y_hat_prior

    return (y_hat_post,
            intercept_post, w_post,
            intercept_C_post, w_C_post, conv_dict)
