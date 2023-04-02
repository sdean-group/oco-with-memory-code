from typing import Callable, List

import numpy as np

def create_rademacher_functions(
    T : int,
    m : int
) -> List[Callable]:
    """
    """

    rng = np.random.default_rng()
    l_sign = rng.choice([-1, 1], size = T, replace = True)

    l_fn = [
        lambda x : l_sign[t] * np.average(x[-m:]) * np.sqrt(m) \
            for t in range(T)
    ]

    return l_fn

def functions_0(
    T : int,
    m : int
) -> List[Callable]:
    """
    """

    rng = np.random.default_rng()

    l_fn = []

    for t in range(T):

        l_coefficient = rng.uniform(-1, 1, size = m)
        def fn(x):
            res = 0
            for i in range(m):
                # res += l_coefficient[i] * (x[-i] - 0.3)**2
                res += (x[-i] - 0.3)**2
            return res

        l_fn.append(fn)

    return l_fn