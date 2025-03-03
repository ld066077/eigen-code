import sympy as sp
import numpy as np
from scipy.optimize import root
import itertools
from sympy import lambdify
from scipy.special import dawsn

# Define additional symbols used later.
M, zeta, Z, Y, tau, coe1_1, coe2_2, q, k = sp.symbols('M zeta Z Y tau coe1_1 coe2_2 q k')
Q0, Q1, Q2, Q3, Q4 = sp.symbols('Q_0 Q_1 Q_2 Q_3 Q_4')
R0, R1, R2, R3, R4 = sp.symbols('R_0 R_1 R_2 R_3 R_4')
Q0 = -Z
R0 = -1/2*Y
Q1 = -zeta+zeta**2*Q0
Q2 = -1/2*zeta+zeta**2*Q1
Q3 = Q2*zeta**2-3/4*zeta
Q4 = Q3*zeta**2-15/8*zeta
R1 = 1/4*(-zeta+zeta**2*R0)
R2 = 1/4*(-zeta/2+zeta**2*R1)
R3 = 1/4*(zeta**2*R2-3/4*zeta)
R4 = 1/4*(zeta**2*R3-15/8*zeta)
lexpr1 = 2*(R1+R0/2)-(Q1+Q0/2)
lexpr2 = 2*(R2+R1+R0/2)-(Q2+Q1+Q0/2)
lexpr3 = 4*(R3+3*R2/2+3*R1/2+3*R0/4)-(Q3+3*Q2/2+3*Q1/2+3*Q0/4)
lexpr4 = 4*(R4+2*R3+3*R2+3*R1+3*R0/2)-(Q4+2*Q3+3*Q2+3*Q1+3*Q0/2)
Gexpr = 1/q**2-1/zeta*(Q2+Q1+Q0/2)-tau*(Q1+Q0/2)**2/(1+tau*(1-zeta*Q0))
Aexpr = 3/(4*q**2)-1/zeta*(Q2+2*Q1+3*Q0/2)-tau*(Q1+Q0/2)*(2*(Q1+Q0)+zeta*Q0*(Q1+Q0/2)/(1+tau*(1-zeta*Q0)))/(1+tau*(1-zeta*Q0))
Bexpr = lexpr4/(2*zeta**3)+tau*lexpr2*lexpr2/(2*zeta**2*(1+tau*(1-zeta*R0)))+tau*(Q1+Q0/2)*( \
    lexpr3/zeta**2 + tau*lexpr1*lexpr2/(zeta*(1+tau*(1-zeta*R0))) + (1-tau**2)*(Q1+Q0/2)*lexpr1*lexpr1/(2*(1+tau*(1-zeta*R0))*(1+tau*(1-zeta*Q0))) + \
        (Q1+Q0/2)*lexpr2/(zeta*(1+tau*(1-zeta*Q0))) + tau*(Q1+Q0/2)*(Q2+Q1+Q0/2)/(2*zeta*(1+tau*(1-zeta*Q0))) )/(1+tau*(1-zeta*Q0))
def F(x):
    """Compute F(x) = -2*dawsn(x) + i*sqrt(pi)*exp(-x^2)色散函数"""
    return -2*dawsn(x) + 1j*np.sqrt(np.pi)*np.exp(-x**2)
def Z_df(zeta_complex):
    Z0 = F(zeta_complex)
    return Z0
expr4_1 = (Gexpr - k**2*(Aexpr+q**2*Bexpr)/2)
symbols_list = [M, zeta, Z, Y, tau, coe1_1, coe2_2, q, k]
expr4_1_np = lambdify(symbols_list, expr4_1, 'numpy')
def Z_df(zeta_complex):
    Z0 = F(zeta_complex)
    return Z0
def expr_to_solve(zeta_val, params):
    zeta_complex = complex(zeta_val[0],zeta_val[1])
    Z_val = Z_df(zeta_complex)
    Y_val = Z_df(zeta_complex/2.0)
    args = (params['M_val'], zeta_complex, Z_val, Y_val, params['tau_val'], params['coe1_1_val'], params['coe2_2_val'], params['q_val'], params['k_val'])
    expr4_1_val = expr4_1_np(*args)

    #print(f"Z_val: {Z_val}, Y_val: {Y_val}")

    return [expr4_1_val.real, expr4_1_val.imag]
def solve_disp_for_params(params):
    zeta_guess = [6.2, 14.2e-03]  # zeta contains q

    #print(f"Initial zeta_guess: {zeta_guess}")
    
    try:
        solution = root(lambda x: expr_to_solve(x,params), zeta_guess)
        #xx=expr_to_solve(solution.x,params)
        #print(xx)
        zeta_solution = complex(solution.x[0], solution.x[1])
        return zeta_solution/params['q_val']
    except Exception as e:
        print(f"Error occurred during finding root:{str(e)}")
        return None
def solve_mutiple_params():
    k_values = [0.1]
    q_values = 1.5 * (1.0 + np.linspace(0, 1, 25))
    M_values = [0.0]
    tau_values = [1.0]
    results = []
    with open('disp_result.txt', 'w') as f:
        f.write('k\tq\tM\ttau\tzeta/q\n')
        for k, q, M, tau in itertools.product(k_values, q_values, M_values, tau_values):
            params = {
                'k_val': k,
                'q_val': q,
                'tau_val': tau,
                'M_val': M,
                'coe1_1_val': 1.0,
                'coe2_2_val': 1.0
            }
            zeta_q = solve_disp_for_params(params)
            if zeta_q is not None:
                f.write(f'{k:.6f}\t{q:.3f}\t{M:.3f}\t{tau:.3f}\t{zeta_q:.5e}\n')
                #f.write(f'{zeta_q:.5e}\n')
                results.append((k, q, M, tau, zeta_q))
        print(f'saved to disp_result.txt')
        return 0
solve_mutiple_params()