import argparse
import csv
import logging

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

def get_step_size(T, A, m):

    if m is None:
        norm_A = np.linalg.norm(A, ord = 2)
        Ltilde = np.sqrt(1.0 / (1 - norm_A**2))
        H2 = np.sqrt((norm_A**2 + norm_A) / (1 - norm_A**2)**3)
        return 1.0 / np.sqrt(T) * 1.0 / np.sqrt(Ltilde * H2 + Ltilde**2)
    else:
        Ltilde = np.sqrt(m)
        H2 = np.sqrt(m**3)
        return 1.0 / np.sqrt(T) * 1.0 / np.sqrt(Ltilde * H2 + Ltilde**2)

def run(T : int, d : int, rho : float, ut_val : float, w : np.array):

    out_prefix = f'output/d_{d}_rho_{rho}_ut_{ut_val}'
    plot_prefix = f'plots/d_{d}_rho_{rho}_ut_{ut_val}'

    # Problem data.
    F = rho * np.eye(d) + np.triu(ut_val * np.ones((d, d)), k = 1)
    G = np.eye(d)
    s_0 = np.zeros(d)
    # F = np.array([[0.9, 0], [0, 0.8]])
    # G = np.array([[1, 0], [1, 0]])
    # # G = G / np.linalg.norm(G, ord = 2)

    # Optimal benchmark.
    logging.info(f'OPT')
    s = cp.Variable((d, T+1))
    u = cp.Variable(d)

    cost = 0
    constraints = []

    constraints += [s[:, 0] == s_0]
    constraints += [cp.norm(u, 2) <= 1]
    for t in range(T):
        cost += cp.sum(s[:, t])
        constraints += [s[:, t+1] == F @ s[:, t] + G @ u + w[t]]

    problem_opt = cp.Problem(cp.Minimize(cost), constraints)
    problem_opt.solve()

    logging.info(f'\t(OPT) Status: {problem_opt.status}')
    logging.info(f'\t(OPT) Cost: {problem_opt.value}')
    logging.info(f'\t(OPT) Control: {u.value}')

    optimal_cost, optimal_control = problem_opt.value, u.value

    # OCO-UM.
    logging.info(f'OCO-UM')
    ftrl_controls = []
    eta = cp.Parameter(nonneg = True)
    eta.value = get_step_size(T, F, m = None)
    logging.info(f'\teta: {eta.value}')

    for k in range(T):

        if k % 10 == 9:
            logging.info(f'\t\t{k+1}/{T}')

        s = cp.Variable((d, k+1))
        u = cp.Variable(d)

        cost = 0
        constraints = []

        constraints += [s[:, 0] == s_0]
        constraints += [cp.norm(u, 2) <= 1]
        cost += cp.norm(u, 2)**2 / eta
        for t in range(k):
            cost += cp.sum(s[:, t])
            constraints += [s[:, t+1] == F @ s[:, t] + G @ u + w[t]]

        problem_oco_um = cp.Problem(cp.Minimize(cost), constraints)
        problem_oco_um.solve()

        if problem_oco_um.status != cp.OPTIMAL:
            logging.info(f'(OCO-UM) Status: {problem_oco_um.status}')
        ftrl_controls.append(u.value)

    # OCO-FM.
    mvals = [1, 4, 8, 16]
    mvals = [1, 16]
    logging.info(f'mvals: {mvals}')

    ftrl_controls_finite = []
    for m in mvals:

        eta = get_step_size(T, F, m)

        logging.info(f'OCO-FM (m = {m})')
        logging.info(f'\teta: {eta}')

        controls = []
        for k in range(T):

            if k % 10 == 9:
                logging.info(f'\t\t{k+1}/{T}')
            
            s = cp.Variable((d, k+1))
            u = cp.Variable(d)

            cost = 0
            constraints = []

            constraints += [s[:, 0] == s_0]
            constraints += [cp.norm(u, 2) <= 1]
            cost += cp.norm(u, 2)**2 / eta
            for t in range(k):
                cost += cp.sum(s[:, t])
                if t <= m:
                    constraints += [s[:, t+1] == F @ s[:, t] + G @ u + w[t]]
                else:
                    term = G @ u + w[t]
                    for i in range(m):
                        term += np.linalg.matrix_power(F, i+1) @ (G @ u + w[t-i-1])
                    constraints += [s[:, t+1] == term]

            problem = cp.Problem(cp.Minimize(cost), constraints)
            problem.solve()

            if problem.status != cp.OPTIMAL:
                logging.info(f'(OCO-FM-{m}) Status: {problem.status}')
            controls.append(u.value)

        ftrl_controls_finite.append(controls)

    # Compute cumulative cost.
    cumulative_cost_opt = []
    cumulative_cost_oco_um = []
    cumulative_cost_oco_fm = [[] for _ in mvals]

    cost = 0
    s = np.array(s_0)
    for t in range(T):
        cost += np.sum(s)
        s = F @ s + G @ optimal_control + w[t]
        cumulative_cost_opt.append(cost)

    cost = 0
    s = np.array(s_0)
    for t in range(T):
        cost += np.sum(s)
        s = F @ s + G @ ftrl_controls[t] + w[t]
        cumulative_cost_oco_um.append(cost)

    for (i, m) in enumerate(mvals):
        cost = 0
        s = np.array(s_0)
        for t in range(T):
            cost += np.sum(s)
            s = F @ s + G @ ftrl_controls_finite[i][t] + w[t]
            cumulative_cost_oco_fm[i].append(cost)

    cumulative_cost_opt = np.array(cumulative_cost_opt)
    cumulative_cost_oco_um = np.array(cumulative_cost_oco_um)
    cumulative_cost_oco_fm = np.array(cumulative_cost_oco_fm)

    logging.info(f'Cumulative Cost.')
    logging.info(f'\tOPT: {cumulative_cost_opt[-1]}')
    logging.info(f'\tOCO-UM: {cumulative_cost_oco_um[-1]}')
    for (i, m) in enumerate(mvals):
        logging.info(f'\tOCO-FM-{m}: {cumulative_cost_oco_fm[i][-1]}')

    # Save cumulative cost.
    with open(f'{out_prefix}_cumulative_cost.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(cumulative_cost_opt)
        writer.writerow(cumulative_cost_oco_um)
        for i in range(len(mvals)):
            writer.writerow(cumulative_cost_oco_fm[i])

    # Save control.
    controls = [
        [optimal_control for _ in range(T)],
        ftrl_controls
    ]
    for i in range(len(mvals)):
        controls.append(ftrl_controls_finite[i])
    np.save(f'{out_prefix}_controls.npy', controls, allow_pickle = True)

    # Plot cumulative cost.
    plt.plot(cumulative_cost_opt, label = 'OPT')
    plt.plot(
        cumulative_cost_oco_um,
        label = 'OCO-UM',
        color = f'C1',
        linestyle = 'dashed'
    )
    for (i, m) in enumerate(mvals):
        plt.plot(
            cumulative_cost_oco_fm[i],
            label = f'OCO-FM-{m}',
            color = f'C{i+2}',
            linestyle = 'dotted'
        )
    plt.xlabel(f'Time')
    plt.ylabel(f'Cumulative Cost')
    plt.title(f'd = {d}, rho = {rho}, ut_val = {ut_val}')
    plt.legend()
    # plt.savefig(f'{plot_prefix}_cumulative_cost.png', bbox_inches='tight')
    plt.clf()
    # plt.show()

    # Plot regret.
    plt.plot(
        cumulative_cost_oco_um - cumulative_cost_opt,
        label = 'OCO-UM',
        color = f'C1',
        linestyle = 'dashed'
    )
    for (i, m) in enumerate(mvals):
        plt.plot(
            cumulative_cost_oco_fm[i] - cumulative_cost_opt,
            label = f'OCO-FM-{m}',
            color = f'C{i+2}',
            linestyle = 'dotted'
        )
    plt.xlabel(f'Time')
    plt.ylabel(f'Regret')
    plt.title(f'd = {d}, rho = {rho}, ut_val = {ut_val}')
    plt.legend()
    plt.savefig(f'{plot_prefix}_regret_opt.png', bbox_inches='tight')
    plt.clf()
    # plt.show()

    # Plot regret relative to FTRL for OCO-UM.
    plt.plot(
        [0] * len(cumulative_cost_oco_um),
        label = 'OCO-UM',
        color = f'C1',
        linestyle = 'dashed'
    )
    for (i, m) in enumerate(mvals):
        plt.plot(
            cumulative_cost_oco_fm[i] - cumulative_cost_oco_um,
            label = f'OCO-FM-{m}',
            color = f'C{i+2}',
            linestyle = 'dotted'
        )
    plt.xlabel(f'Time')
    plt.ylabel(f'Regret vs FTRL for OCO-UM')
    plt.title(f'd = {d}, rho = {rho}, ut_val = {ut_val}')
    plt.legend()
    # plt.savefig(f'{plot_prefix}_regret_ocoum.png', bbox_inches='tight')
    plt.clf()
    # plt.show()

def main():
    
    plt.style.use('fivethirtyeight')

    logging.basicConfig(
        format = '%(asctime)s %(filename)s:%(lineno)d %(message)s',
        datefmt = '%I:%M:%S %p',
        level = logging.INFO
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--d',
                        type = int,
                        default = 2,
                        help = 'dimension.')
    parser.add_argument('--rho',
                        type = float,
                        default = 0.9,
                        help = 'value for the diagonal of F.')
    parser.add_argument('--upper_triangular_val',
                        type = float,
                        default = 0,
                        help = 'value for the upper triangular part of F.')
    parser.add_argument('--T',
                        type = int,
                        default = 100,
                        help = 'time horizon.')
    parser.add_argument('--use_same_dist',
                        action = 'store_true',
                        help = 'Use the same disturbance for all parameters.')
    args = parser.parse_args()

    rng = np.random.default_rng()
    w = rng.normal(size = (args.T, args.d))

    hyperparameters= [
        (0.9, [1, 5, 10]),
        # (0.9, [0, 0.1, 0.15, 0.18]),
        # (0.95, [0, 0.05, 0.09])
    ]

    if args.use_same_dist:
        for (rho, ut_vals) in hyperparameters:
            for ut_val in ut_vals:
                logging.info(f'Running with rho = {rho}, ut_val = {ut_val}')
                run(args.T, args.d, rho, ut_val, w)
    else:
        run(args.T, args.d, args.rho, args.upper_triangular_val, w)

if __name__ == '__main__':
    main()