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

    l_fn = []
    for t in range(T):

        def fn(x : List[float]):
            """
            """

            return l_sign[t] * np.average(x[-m:]) * np.sqrt(m)

        l_fn.append(fn)

    return l_fn