import numpy as np

from karma.core.utils import create_timer
from karma.learning.logistic import CONVERGENCE_INFO_DESIGN_WIDTH
from karma.learning.utils import BasicVirtualHStack
from karma.learning.matrix_utils import diagonal_of_inverse_symposdef
from karma.learning.regression import linear_regression_coefficients


def linear_coefficients_and_posteriori(X, y, w_priori=None, intercept_priori=0.,
                                       C_priori=1e5, intercept_C_priori=1e10,
                                       sample_weight=None, hessian_mode='full',
                                       noise_variance=1., timer=None):
    if timer is None:
        timer = create_timer(None)

    conv_dict = {}

    with timer('BayLinReg_Reg_Init'):
        if not isinstance(X, BasicVirtualHStack):
            X = BasicVirtualHStack(X)
        conv_dict[CONVERGENCE_INFO_DESIGN_WIDTH] = X.shape[-1]
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
        if hessian_mode == 'skip':
            C_post = np.full(X.shape[1] + 1, np.nan, dtype=np.float)
        elif hessian_mode == 'full':
            inv_cov = np.diag(1. / C_priori)
            inv_cov += XX.transpose().dot(XX) * noise_precision
            C_post = diagonal_of_inverse_symposdef(inv_cov)
        elif hessian_mode == 'diag':
            C_post = 1. / (1. / C_priori + np.einsum('ij,ij->j', XX, XX) * noise_precision)
        else:
            raise ValueError('hessian_mode needs to be one of {{skip, full, diag}}, got {}'.format(hessian_mode))


        intercept_C_post, feature_C_post = C_post[-1], C_post[:-1]
        feature_C_posts = X.split_by_dims(feature_C_post.astype(np.float))

    y_hat_post = y_hat_post_prime + y_hat_prior

    return (y_hat_post,
            intercept_post, w_post,
            intercept_C_post, feature_C_posts, conv_dict)
