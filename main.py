import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np

import algorithms
import loss_functions

# Implement other loss functions.
# Experiment.

def main():

    plt.style.use('ggplot')

    logging.basicConfig(
        format = '%(asctime)s %(filename)s:%(lineno)d %(message)s',
        datefmt = '%I:%M:%S %p',
        level = logging.INFO
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--m',
                        type = int,
                        default = 2,
                        help = 'memory length.')
    parser.add_argument('--T',
                        type = int,
                        default = 100,
                        help = 'time horizon.')
    args = parser.parse_args()

    rng = np.random.default_rng()

    # Create linear functions whose coefficients are Rademacher random
    # variables.
    l_fn = loss_functions.create_rademacher_functions(args.T, args.m)

    # Find the benchmark using scipy.optimize.
    opt = algorithms.optimal_benchmark(
        l_fn,
        rng.random(size = 1),
        args.T,
        args.m
    )
    cum_loss_opt = np.cumsum([l_fn[t](opt.x) for t in range(args.T)])

    # FTRL.
    l_ftrl_res, l_regret = [], []

    for mode in algorithms.mode_strs:

        init_fn = getattr(algorithms, f'ftrl_init_{mode}')
        z_0 = init_fn(args.m)

        eta = None
        eta = 1 / np.sqrt(args.T)

        l_ftrl_res.append(algorithms.ftrl(
            l_fn,
            z_0,
            args.T,
            args.m,
            eta,
            mode
        ))

        cum_loss_alg = np.cumsum([res.l for res in l_ftrl_res[-1]])
        l_regret.append(cum_loss_alg - cum_loss_opt)

    # Plot regret.
    for i, mode in enumerate(algorithms.mode_strs):
        label = algorithms.mode_strs[mode]
        plt.plot(l_regret[i], label = label)
    plt.xlabel(f'Time t')
    plt.ylabel(f'Regret')
    plt.legend()
    plt.show()

    # Plot: \| h_t - (x_t, \dots, x_t) \|_2.
    for i, mode in enumerate(algorithms.mode_strs):
        label = algorithms.mode_strs[mode]
        label = mode

        l_diffs = []

        for t in range(args.T):

            # Construct h_t using the last m elements of the history and pad the
            # prefix with 0 if necessary.
            h_t = [0] * max(0, args.m - t - 1) + \
                [l_ftrl_res[i][s].x for s in range(max(0, t - args.m + 1), t+1)]
            
            # Compute \| h_t - (x_t, \dots, x_t) \|_2.
            diff = np.linalg.norm(
                np.array(h_t) - np.array([l_ftrl_res[i][t].x] * len(h_t))
            )

            l_diffs.append(diff)

        plt.plot(l_diffs, label = label)
        logging.info(f'Sum of norm differences for mode {mode}: {sum(l_diffs)}')
    plt.xlabel(f'Time t')
    plt.ylabel(f'\| h_t - (x_t, \dots, x_t) \|_2^2')
    plt.legend()
    plt.show()

    # Plot norm of the difference between the iterates on modes 0 and 2.
    l_diffs = [
        np.linalg.norm(l_ftrl_res[0][i].x - l_ftrl_res[2][i].x) \
            for i in range(args.T)
    ]
    plt.plot(l_diffs)
    logging.info(f'Sum of norm differences for modes 0 and 2: {sum(l_diffs)}')
    plt.xlabel(f'Time t')
    plt.ylabel(f'Norm of the difference between the iterates on modes 0 and 2')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()