from typing import Callable, List

import numpy as np

def average_of_past_decisions_with_rademacher_signs(
    T : int,
    m : int
) -> List[Callable]:
    """
    f_t(h_t) = w_t * 1/sqrt(m) * \sum_{k=0}^{m-1} x_{t-k}, where w_t is a
    Rademacher random variable.
    """

    rng = np.random.default_rng()
    l_sign = rng.choice([-1, 1], size = T, replace = True)

    l_fn = [
        lambda x : l_sign[t] * np.average(x[-m:]) * np.sqrt(m) \
            for t in range(T)
    ]

    return l_fn

def square_diff_around_fixed_point(
    T : int,
    m : int
) -> List[Callable]:
    """
    f_t(h_t) = \sum_{k=0}^{m-1} (x_{t-k} - 0.3)^2
    """

    l_fn = [
        lambda x : sum([(x[-i] - 0.3)**2 for i in range(m)]) \
            for t in range(T)
    ]

    return l_fn