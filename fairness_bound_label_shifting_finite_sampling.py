import numpy as np
from scipy.optimize import fmin
from scipy.optimize import fmin_tnc
import math
import scipy.optimize


def fairness_certification_bound_label_shifting_finite_sampling(hellinger_distances,lambda_P, E_P, gamma_k=0.0,gamma_r=0.0):
    def fairness_grammain(x):
        return -E_P[0] * x[0] * x[1] - E_P[1] * x[0] * (1 - x[1]) - E_P[2] * (1 - x[0]) * x[1] - E_P[3] * (
                    1 - x[0]) * (1 - x[1])

    epsilon = 1e-6
    results = []
    for dist in hellinger_distances:
        dis = dist
        result = scipy.optimize.minimize(fairness_grammain,
                                         np.array([0.5, 0.5]),
                                         bounds=[(epsilon+gamma_k, 1-epsilon-gamma_k), (epsilon+gamma_r, 1-epsilon-gamma_r)],
                                         constraints=[
                                             {'type': 'ineq', 'fun': lambda x: math.sqrt(lambda_P[0] * x[0] * x[1]) + math.sqrt(lambda_P[1] * x[0] * (
                                                     1 - x[1])) +
                                         math.sqrt(lambda_P[2] * (1 - x[0]) * x[1]) + math.sqrt(lambda_P[3] * (
                                                     1 - x[0]) * (1 - x[1])) - 1 + dis ** 2 + epsilon}]
                                         )
        upper_bnd = -fairness_grammain(result.x)
        results.append(upper_bnd)
    return results

def fairness_certification_bound_label_shifting_finite_sampling_improve(hellinger_distances,lambda_P, E_P, gamma_k=0.0,gamma_r=0.0):
    def fairness_grammain(x):
        return -E_P[0] * x[0] * x[1] - E_P[1] * x[0] * (1 - x[1]) - E_P[2] * (1 - x[0]) * x[1] - E_P[3] * (
                    1 - x[0]) * (1 - x[1])

    epsilon = 1e-6
    results = []
    tot = sum(lambda_P)
    for dist in hellinger_distances:
        dis = dist
        result = scipy.optimize.minimize(fairness_grammain,
                                         np.array([lambda_P[0]+lambda_P[1],lambda_P[0]+lambda_P[2],lambda_P[0]/tot,lambda_P[1]/tot,lambda_P[2]/tot,lambda_P[3]/tot]),
                                         bounds=[(epsilon+gamma_k, 1-epsilon-gamma_k), (epsilon+gamma_r, 1-epsilon-gamma_r),
                                                 (0,lambda_P[0]),(0,lambda_P[1]),(0,lambda_P[2]),(0,lambda_P[3]),],
                                         constraints=[
                                             {'type': 'ineq', 'fun': lambda x: math.sqrt(x[2] * x[0] * x[1]) + math.sqrt(x[3] * x[0] * (
                                                     1 - x[1])) +
                                         math.sqrt(x[4] * (1 - x[0]) * x[1]) + math.sqrt(x[5] * (
                                                     1 - x[0]) * (1 - x[1])) - 1 + dis ** 2},
                                             {'type': 'eq', 'fun': lambda x: x[2]+x[3]+x[4]+x[5]-1}]
                                         )
        upper_bnd = -fairness_grammain(result.x)
        results.append(upper_bnd)
    return results
