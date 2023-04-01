import logging
from typing import Callable, List

import numpy as np
import scipy

from custom_types import FTRLResult

mode_strs = {
    0: 'FTRL on f_t and project by taking the average.',
    1: 'FTRL on f_t and project onto the most recent decision.',
    2: 'FTRL on \circfn f_t.',
    3: 'FTRL on f_{1:t}((x, x_{t-1}, \dots, x_{t-m+1})).',
    4: 'FTRL on \circfn f_t plus \| (x, \dots, x) - (x, x_{t-1}, \dots, x_{t-m+1}) \|_2^2.'
}

def optimal_benchmark(
    l_fn : List[Callable],
    x_0 : float,
    T : int,
    m : int
) -> scipy.optimize.OptimizeResult:

    logging.info(f'Optimal benchmark.')

    obj = lambda x : sum([l_fn[t]([x] * m) for t in range(T)])
    res = scipy.optimize.minimize(
        obj,
        x_0,
        bounds = [(-1, 1)]
    )

    logging.info(f'\tx: {res.x}')
    logging.info(f'\tsuccess: {res.success}')
    logging.info(f'\tmessage: {res.message}')
    logging.info(f'\tfun: {res.fun}')

    return res

def ftrl(
    l_fn : List[Callable],
    z_0 : List[float],
    T : int,
    m : int,
    eta : float,
    mode : int
) -> List[FTRLResult]:
    """
    Mode 0: FTRL on f_t and project by taking the average.
    Mode 1: FTRL on f_t and project onto the most recent decision.
    Mode 2: FTRL on \circfn f_t.
    Mode 3: FTRL on f_{1:t}((x, x_{t-1}, \dots, x_{t-m+1})).
    Mode 4: FTRL on \circfn f_t plus \| (x, \dots, x) - (x, x_{t-1}, \dots, x_{t-m+1}) \|_2^2.
    """

    optimize_over_histories = set([0, 1])
    optimize_over_decisions = set([2, 3, 4])

    if mode not in mode_strs:
        raise ValueError(f'Unknown mode {mode}.')
    logging.info(f'mode {mode}: {mode_strs[mode]}')

    if mode in optimize_over_histories:
        cur_z = z_0
    elif mode in optimize_over_decisions:
        cur_z = z_0[0]
    l_ftrl_res = []
    history = [0] * m

    for t in range(T):

        # Define objective function.
        obj = globals()[f'ftrl_obj_{mode}'](
            l_fn = l_fn,
            t = t,
            eta = eta,
            m = m,
            history = history
        )

        # Optimize.
        if mode in optimize_over_histories:
            bounds = [(-1, 1)] * m
        else:
            bounds = [(-1, 1)]
        res = scipy.optimize.minimize(
            obj,
            cur_z,
            bounds = bounds
        )

        logging.debug(f'\t({t}/{T})z: {res.x}')
        logging.debug(f'\t({t}/{T})success: {res.success}')
        logging.debug(f'\t({t}/{T})message: {res.message}')
        logging.debug(f'\t({t}/{T})fun: {res.fun}')

        # Update intial guess for next iteration.
        if mode in optimize_over_histories:
            cur_z = res.x
        elif mode in optimize_over_decisions:
            cur_z = res.x[0]

        # Obtain decision from optimization result.
        x_t = globals()[f'ftrl_decision_{mode}'](
            z = res.x
        )

        history.append(x_t)
        l_t = l_fn[t](history[-m : ])
        l_ftrl_res.append(FTRLResult(x_t, l_t))

    return l_ftrl_res

def ftrl_obj_0(
    l_fn : List[Callable],
    t : int,
    eta : float,
    m : int,
    history : List[float]
) -> Callable:
    """
    Mode 0: FTRL on f_t and project by taking the average.
    """

    loss = lambda z : sum([l_fn[i](z) for i in range(t)])
    regularizer = lambda z : sum([np.linalg.norm(x)**2 for x in z])
    obj = lambda z : loss(z) + regularizer(z) / eta

    return obj

def ftrl_obj_1(
    l_fn : List[Callable],
    t : int,
    eta : float,
    m : int,
    history : List[float]
) -> Callable:
    """
    Mode 1: FTRL on f_t and project onto the most recent decision.
    """

    return ftrl_obj_0(l_fn, t, eta, m, history)

def ftrl_obj_2(
    l_fn : List[Callable],
    t : int,
    eta : float,
    m : int,
    history : List[float]
) -> Callable:
    """
    Mode 2: FTRL on \circfn f_t.
    """

    loss = lambda z : sum([l_fn[i]([z] * m) for i in range(t)])
    regularizer = lambda z : np.linalg.norm(z)**2
    obj = lambda z : loss(z) + regularizer(z) / eta

    return obj

def ftrl_obj_3(
    l_fn : List[Callable],
    t : int,
    eta : float,
    m : int,
    history : List[float]
) -> Callable:
    """
    Mode 3: FTRL on f_{1:t}((x, x_{t-1}, \dots, x_{t-m+1})).
    """

    loss = lambda z : sum([
        l_fn[i](z + history[-m+1:][::-1]) for i in range(t)
    ])
    regularizer = lambda z : np.linalg.norm(z)**2
    obj = lambda z : loss(z) + regularizer(z) / eta

    return obj

def ftrl_obj_4(
    l_fn : List[Callable],
    t : int,
    eta : float,
    m : int,
    history : List[float]
) -> Callable:
    """
    Mode 4: FTRL on \circfn f_t plus \| (x, \dots, x) - (x, x_{t-1}, \dots, x_{t-m+1}) \|_2^2.
    """

    loss = lambda z : sum([l_fn[i]([z] * m) for i in range(t)])
    regularizer_1 = lambda z : np.linalg.norm(z)**2
    regularizer_2 = lambda z : np.linalg.norm(
        np.array([z] * m) - np.array(history[-m+1:][::-1] + z)
    )**2
    obj = lambda z : loss(z) + regularizer_1(z) / eta + regularizer_2(z) / eta

    return obj

def ftrl_decision_0(
    z : List[float]
) -> float:
    """
    Mode 0: FTRL on f_t and project by taking the average.
    """

    return np.average(z)

def ftrl_decision_1(
    z : List[float]
) -> float:
    """
    Mode 1: FTRL on f_t and project onto the most recent decision.
    """
    
    return z[0]

def ftrl_decision_2(
    z : List[float]
) -> float:
    """
    Mode 2: FTRL on \circfn f_t.
    """

    return z[0]

def ftrl_decision_3(
    z : List[float]
) -> float:
    """
    Mode 3: FTRL on f_{1:t}((x, x_{t-1}, \dots, x_{t-m+1})).
    """

    return z[0]

def ftrl_decision_4(
    z : List[float]
) -> float:
    """
    Mode 4: FTRL on \circfn f_t plus \| (x, \dots, x) - (x, x_{t-1}, \dots, x_{t-m+1}) \|_2^2.
    """

    return z[0]

def ftrl_init_0(m : int):
    """
    Mode 0: FTRL on f_t and project by taking the average.
    """

    return np.random.default_rng().random(size = m)

def ftrl_init_1(m : int):
    """
    Mode 1: FTRL on f_t and project onto the most recent decision.
    """

    return np.random.default_rng().random(size = m)

def ftrl_init_2(m : int):
    """
    Mode 2: FTRL on \circfn f_t.
    Mode 3: FTRL on f_{1:t}((x, x_{t-1}, \dots, x_{t-m+1})).
    Mode 4: FTRL on \circfn f_t plus \| (x, \dots, x) - (x, x_{t-1}, \dots, x_{t-m+1}) \|_2^2.
    """

    return np.random.default_rng().random(size = 1)

def ftrl_init_3(m : int):
    """
    Mode 3: FTRL on f_{1:t}((x, x_{t-1}, \dots, x_{t-m+1})).
    Mode 4: FTRL on \circfn f_t plus \| (x, \dots, x) - (x, x_{t-1}, \dots, x_{t-m+1}) \|_2^2.
    """

    return np.random.default_rng().random(size = 1)

def ftrl_init_4(m : int):
    """
    Mode 4: FTRL on \circfn f_t plus \| (x, \dots, x) - (x, x_{t-1}, \dots, x_{t-m+1}) \|_2^2.
    """

    return np.random.default_rng().random(size = 1)