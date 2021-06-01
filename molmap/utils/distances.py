import numpy as np
import numba



################### numeric data #########################
@numba.njit(fastmath=True)
def euclidean(x, y):
    """Standard euclidean distance. l2 distance
    ..math::
        D(x, y) = \sqrt{\sum_i (x_i - y_i)^2}
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (x[i] - y[i]) ** 2
    return np.sqrt(result)

@numba.njit(fastmath=True)
def sqeuclidean(x, y):
    """Standard euclidean distance. l2 distance
    ..math::
        D(x, y) = \sqrt{\sum_i (x_i - y_i)^2}
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (x[i] - y[i]) ** 2
    return result



@numba.njit()
def manhattan(x, y):
    """Manhatten, taxicab, or l1 distance.
    ..math::
        D(x, y) = \sum_i |x_i - y_i|
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += np.abs(x[i] - y[i])

    return result


@numba.njit()
def canberra(x, y):
    result = 0.0
    for i in range(x.shape[0]):
        denominator = np.abs(x[i]) + np.abs(y[i])
        if denominator > 0:
            result += np.abs(x[i] - y[i]) / denominator

    return result



@numba.njit()
def chebyshev(x, y):
    """Chebyshev or l-infinity distance.
    ..math::
        D(x, y) = \max_i |x_i - y_i|
    """
    result = 0.0
    for i in range(x.shape[0]):
        result = max(result, np.abs(x[i] - y[i]))

    return result



############### binary data ################
@numba.njit()
def jaccard(x, y):
    num_non_zero = 0.0
    num_equal = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_non_zero += x_true or y_true
        num_equal += x_true and y_true

    if num_non_zero == 0.0:
        return 0.0
    else:
        return float(num_non_zero - num_equal) / num_non_zero
    
@numba.njit()
def rogers_tanimoto(x, y):
    num_not_equal = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_not_equal += x_true != y_true

    return (2.0 * num_not_equal) / (x.shape[0] + num_not_equal)



@numba.njit()
def hamming(x, y):
    result = 0.0
    for i in range(x.shape[0]):
        if x[i] != y[i]:
            result += 1.0

    return float(result) / x.shape[0]


@numba.njit()
def dice(x, y):
    num_true_true = 0.0
    num_not_equal = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_true_true += x_true and y_true
        num_not_equal += x_true != y_true

    if num_not_equal == 0.0:
        return 0.0
    else:
        return num_not_equal / (2.0 * num_true_true + num_not_equal)


@numba.njit()
def kulsinski(x, y):
    num_true_true = 0.0
    num_not_equal = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_true_true += x_true and y_true
        num_not_equal += x_true != y_true

    if num_not_equal == 0:
        return 0.0
    else:
        return float(num_not_equal - num_true_true + x.shape[0]) / (
            num_not_equal + x.shape[0]
        )
    
@numba.njit()
def sokal_sneath(x, y):
    num_true_true = 0.0
    num_not_equal = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_true_true += x_true and y_true
        num_not_equal += x_true != y_true

    if num_not_equal == 0.0:
        return 0.0
    else:
        return num_not_equal / (0.5 * num_true_true + num_not_equal)

    

    
################### both #############
@numba.njit()
def bray_curtis(x, y):
    numerator = 0.0
    denominator = 0.0
    for i in range(x.shape[0]):
        numerator += np.abs(x[i] - y[i])
        denominator += np.abs(x[i] + y[i])

    if denominator > 0.0:
        return float(numerator) / denominator
    else:
        return 0.0


@numba.njit()
def cosine(x, y):
    result = 0.0
    norm_x = 0.0
    norm_y = 0.0
    for i in range(x.shape[0]):
        result += x[i] * y[i]
        norm_x += x[i] ** 2
        norm_y += y[i] ** 2

    if norm_x == 0.0 and norm_y == 0.0:
        return 0.0
    elif norm_x == 0.0 or norm_y == 0.0:
        return 1.0
    else:
        return 1.0 - (result / np.sqrt(norm_x * norm_y))


@numba.njit()
def correlation(x, y):
    mu_x = 0.0
    mu_y = 0.0
    norm_x = 0.0
    norm_y = 0.0
    dot_product = 0.0

    for i in range(x.shape[0]):
        mu_x += x[i]
        mu_y += y[i]

    mu_x /= x.shape[0]
    mu_y /= x.shape[0]

    for i in range(x.shape[0]):
        shifted_x = x[i] - mu_x
        shifted_y = y[i] - mu_y
        norm_x += shifted_x ** 2
        norm_y += shifted_y ** 2
        dot_product += shifted_x * shifted_y

    if norm_x == 0.0 and norm_y == 0.0:
        return 0.0
    elif dot_product == 0.0:
        return 1.0
    else:
        return 1.0 - (dot_product / np.sqrt(norm_x * norm_y))
    
    
    
descriptors_dist = [(euclidean,'euclidean'),
                    (sqeuclidean,'sqeuclidean'),
                    (manhattan,'manhattan'),
                    (canberra,'canberra'),
                    (chebyshev,'chebyshev'),
                    (cosine,'cosine'),
                    (correlation,'correlation'),
                    (bray_curtis,'braycurtis')]



fingerprint_dist = [(jaccard, 'jaccard'),
                    (rogers_tanimoto, 'rogers_tanimoto'),
                    (hamming,'hamming'),
                    (dice, 'dice'),
                    (kulsinski, 'kulsinski'),
                    (sokal_sneath,'sokal_sneath'),
                    (cosine,'cosine'),
                    (correlation,'correlation'),
                    (bray_curtis,'braycurtis')]




def GenNamedDist(descriptors_dist, fingerprint_dist):
    _dist_fuc = {}
    _all = descriptors_dist.copy()
    _all.extend(fingerprint_dist)
    for i in _all:
        _dist_fuc[i[1]] = i[0]
    return _dist_fuc


named_distances = GenNamedDist(descriptors_dist, fingerprint_dist)




if __name__ == '__main__':
    
    import pandas as pd
    
    x = np.random.random_sample(size=(100,2))
    x1 = x.round()
    
    res = {}
    for f,k in descriptors_dist:
        ks = 'descriptors-' + k
        res.update({ks:f(x[:,0], x[:,1])})
        
    for f,k in fingerprint_dist:
        ks = 'fingerprint-' + k
        res.update({ks :f(x1[:,0], x1[:,1])})   
        
    
    print(pd.Series(res))
