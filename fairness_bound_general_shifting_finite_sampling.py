import numpy as np
from scipy.optimize import fmin
import math
import scipy.optimize
from tqdm import tqdm


M = 1
K, R = 0.5, 0.5
P_00, P_01, P_10, P_11 = 0.25,0.25,0.25,0.25
E_00_up, E_01_up, E_10_up, E_11_up = 0,0,0,0
E_00_low, E_01_low, E_10_low, E_11_low = 0,0,0,0
V_00_up, V_01_up, V_10_up, V_11_up = 0,0,0,0
V_00_low, V_01_low, V_10_low, V_11_low = 0,0,0,0

def fairness_grammian_improve(x):
    global M, K, R, P_00, P_01, P_10, P_11, E_00_up, E_01_up, E_10_up, E_11_up, E_00_low, E_01_low, E_10_low, E_11_low, \
        V_00_up, V_01_up, V_10_up, V_11_up, V_00_low, V_01_low, V_10_low, V_11_low, k_up, k_low, r_up, r_low
    # due to the coefficients of E_00 > 0, consider the coefficients before
    C_00_up = M - E_00_up - V_00_low / (M - E_00_low)
    C_01_up = M - E_01_up - V_01_low / (M - E_01_low)
    C_10_up = M - E_10_up - V_10_low / (M - E_10_low)
    C_11_up = M - E_11_up - V_11_low / (M - E_11_low)

    C_00_low = M - E_00_up - V_00_up / (M - E_00_up)
    C_01_low = M - E_01_up - V_01_up / (M - E_01_up)
    C_10_low = M - E_10_up - V_10_up / (M - E_10_up)
    C_11_low = M - E_11_up - V_11_up / (M - E_11_up)

    res = max(k_up*r_up*(E_00_up+C_00_up),k_low*r_low*(E_00_up+C_00_up)) + max(k_up*(1-r_low)*(E_01_up+C_01_up),k_low*(1-r_up)*(E_01_up+C_01_up)) \
         + max((1-k_low)*r_up*(E_10_up+C_10_up),(1-k_up)*r_low*(E_10_up+C_10_up)) + max((1-k_up)*(1-r_up)*(E_11_up+C_11_up),(1-k_low)*(1-r_low)*(E_11_up+C_11_up)) \
         - x[0]*min(k_low*r_low*C_00_low,k_up*r_up*C_00_low) - x[1]*min(k_low*(1-r_up)*C_01_low,k_up*(1-r_low)*C_01_low) \
         - x[2]*min((1-k_low)*r_up*C_10_low,(1-k_up)*r_low*C_10_low) - x[3]*min((1-k_low)*(1-r_low)*C_11_low,(1-k_up)*(1-r_up)*C_11_low) \
         + 2*k_up*r_up*math.sqrt(x[0]*(1-x[0])*V_00_up) + 2*k_up*(1-r_low)*math.sqrt(x[1]*(1-x[1])*V_01_up) \
         + 2*(1-k_low)*r_up*math.sqrt(x[2]*(1-x[2])*V_10_up) + 2*(1-k_low)*(1-r_low)*math.sqrt(x[3]*(1-x[3])*V_11_up)

    return -res


def fairness_upper_bound_general_shifting_finite_sampling(hellinger_distances, lambda_P, E_P_up, E_P_low, V_P_up, V_P_low, gamma_k, gamma_r, interval_kr, lambda_P_low, lambda_P_up):
    global M, K, R, P_00, P_01, P_10, P_11, E_00_up, E_01_up, E_10_up, E_11_up, E_00_low, E_01_low, E_10_low, E_11_low, \
        V_00_up, V_01_up, V_10_up, V_11_up, V_00_low, V_01_low, V_10_low, V_11_low, k_up, k_low, r_up, r_low
    if gamma_k == 0.5:
        num_intervals_k = 1
    else:
        num_intervals_k = int((1-2*gamma_k)/interval_kr)
    if gamma_r == 0.5:
        num_intervals_r = 1
    else:
        num_intervals_r = int((1 - 2 * gamma_r) / interval_kr)
    gamma = 0.0
    P_00, P_01, P_10, P_11 = lambda_P[0], lambda_P[1], lambda_P[2], lambda_P[3]
    E_00_up, E_01_up, E_10_up, E_11_up = E_P_up[0], E_P_up[1], E_P_up[2], E_P_up[3]
    E_00_low, E_01_low, E_10_low, E_11_low = E_P_low[0], E_P_low[1], E_P_low[2], E_P_low[3]
    V_00_up, V_01_up, V_10_up, V_11_up = V_P_up[0], V_P_up[1], V_P_up[2], V_P_up[3]
    V_00_low, V_01_low, V_10_low, V_11_low = V_P_low[0], V_P_low[1], V_P_low[2], V_P_low[3]

    upper_bounds = []
    for dist in tqdm(hellinger_distances):
        upper = 0
        upper_k = -1
        upper_r = -1
        upper_x = None
        for k in tqdm(np.linspace(gamma_k,1-gamma_k-interval_kr,num_intervals_k)):
            for r in np.linspace(gamma_r,1-gamma_r-interval_kr,num_intervals_r):
                K = k
                R = r
                epsilon = 0.0001

                k_up = k + interval_kr
                k_low = k
                r_up = r + interval_kr
                r_low = r

                if math.sqrt(k_up*r_up*P_00) + math.sqrt(k_up*(1-r_low)*P_01) \
                              + math.sqrt((1-k_low)*r_up*P_10) + math.sqrt((1-k_low)*(1-r_low)*P_11) -1 + dist**2 < epsilon:
                    continue

                lower_bnd_of_x = []
                for i in range(4):
                    rho_up = math.sqrt(1-(1+(M-E_P_low[i])**2/V_P_low[i])**(-0.5))
                    lower_bnd_of_x.append((1-rho_up**2)**2)

                result = scipy.optimize.minimize(fairness_grammian_improve,
                         np.array([1-epsilon,1-epsilon,1-epsilon,1-epsilon,lambda_P[0],lambda_P[1],lambda_P[2],lambda_P[3]]),
                         bounds=[(lower_bnd_of_x[0]+epsilon,1-epsilon),
                                 (lower_bnd_of_x[1]+epsilon,1-epsilon),
                                 (lower_bnd_of_x[2]+epsilon,1-epsilon),
                                 (lower_bnd_of_x[3]+epsilon,1-epsilon),
                                 (lambda_P_low[0]-epsilon,lambda_P_up[0]+epsilon),
                                 (lambda_P_low[1]-epsilon,lambda_P_up[1]+epsilon),
                                 (lambda_P_low[2]-epsilon,lambda_P_up[2]+epsilon),
                                 (lambda_P_low[3]-epsilon,lambda_P_up[3]+epsilon)],
                         constraints=[
                             {'type': 'ineq',
                              'fun': lambda x: math.sqrt(k_up*r_up*P_00*x[0]) + math.sqrt(k_up*(1-r_low)*P_01*x[1]) \
                              + math.sqrt((1-k_low)*r_up*P_10*x[2]) + math.sqrt((1-k_low)*(1-r_low)*P_11*x[3]) -1 + dist**2 -epsilon},
                             {'type': 'eq',
                              'fun': lambda x: x[4]+x[5]+x[6]+x[7]-1}
                         ])

                upper_tmp = -fairness_grammian_improve(result.x)
                if upper_tmp > upper:
                    upper = upper_tmp
                    upper_k = k
                    upper_r = r
                    upper_x=result.x

        upper_bounds.append(upper)
        # print(f'{dist}: {upper}  k and r are: {upper_k}  {upper_r}  solution x is: {upper_x}')
    return upper_bounds