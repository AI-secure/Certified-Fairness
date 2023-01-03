import numpy as np
from scipy.optimize import fmin
import math
import scipy.optimize
from tqdm import tqdm


M = 1
K, R = 0.5, 0.5
P_00, P_01, P_10, P_11 = 0.25,0.25,0.25,0.25
E_00, E_01, E_10, E_11 = 0,0,0,0
V_00, V_01, V_10, V_11 = 0,0,0,0

def fairness_grammian_improve(x):
    global M, K, R, P_00, P_01, P_10, P_11, E_00, E_01, E_10, E_11,  \
        V_00, V_01, V_10, V_11, k_up, k_low, r_up, r_low
    C_00 = M - E_00 - V_00 / (M - E_00)
    C_01 = M - E_01 - V_01 / (M - E_01)
    C_10 = M - E_10 - V_10 / (M - E_10)
    C_11 = M - E_11 - V_11 / (M - E_11)

    res = max(k_up*r_up*(E_00+C_00),k_low*r_low*(E_00+C_00)) + max(k_up*(1-r_low)*(E_01+C_01),k_low*(1-r_up)*(E_01+C_01)) \
         + max((1-k_low)*r_up*(E_10+C_10),(1-k_up)*r_low*(E_10+C_10)) + max((1-k_up)*(1-r_up)*(E_11+C_11),(1-k_low)*(1-r_low)*(E_11+C_11)) \
         - x[0]*min(k_low*r_low*C_00,k_up*r_up*C_00) - x[1]*min(k_low*(1-r_up)*C_01,k_up*(1-r_low)*C_01) \
         - x[2]*min((1-k_low)*r_up*C_10,(1-k_up)*r_low*C_10) - x[3]*min((1-k_low)*(1-r_low)*C_11,(1-k_up)*(1-r_up)*C_11) \
         + 2*k_up*r_up*math.sqrt(x[0]*(1-x[0])*V_00) + 2*k_up*(1-r_low)*math.sqrt(x[1]*(1-x[1])*V_01) \
         + 2*(1-k_low)*r_up*math.sqrt(x[2]*(1-x[2])*V_10) + 2*(1-k_low)*(1-r_low)*math.sqrt(x[3]*(1-x[3])*V_11)

    return -res


def fairness_upper_bound_general_shifting(hellinger_distances, lambda_P, E_P, V_P, gamma_k, gamma_r, interval_kr):
    global M, K, R, P_00, P_01, P_10, P_11, E_00, E_01, E_10, E_11, \
        V_00, V_01, V_10, V_11, k_up, k_low, r_up, r_low
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
    E_00, E_01, E_10, E_11 = E_P[0], E_P[1], E_P[2], E_P[3]
    V_00, V_01, V_10, V_11 = V_P[0], V_P[1], V_P[2], V_P[3]

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
                    rho_up = math.sqrt(1-(1+(M-E_P[i])**2/V_P[i])**(-0.5))
                    lower_bnd_of_x.append((1-rho_up**2)**2)

                result = scipy.optimize.minimize(fairness_grammian_improve,
                         np.array([1-epsilon,1-epsilon,1-epsilon,1-epsilon]),
                         bounds=[(lower_bnd_of_x[0]+epsilon,1-epsilon),
                                 (lower_bnd_of_x[1]+epsilon,1-epsilon),
                                 (lower_bnd_of_x[2]+epsilon,1-epsilon),
                                 (lower_bnd_of_x[3]+epsilon,1-epsilon),
                                 ],
                         constraints=[
                             {'type': 'ineq',
                              'fun': lambda x: math.sqrt(k_up*r_up*P_00*x[0]) + math.sqrt(k_up*(1-r_low)*P_01*x[1]) \
                              + math.sqrt((1-k_low)*r_up*P_10*x[2]) + math.sqrt((1-k_low)*(1-r_low)*P_11*x[3]) -1 + dist**2 -epsilon},

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