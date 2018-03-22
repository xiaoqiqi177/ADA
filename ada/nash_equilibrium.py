#!/usr/bin/env python
# coding: utf-8
import numpy as np
from solve_lp_gurobi import solve_f, solve_p
from losses import *

def solveGame(dets, Sf, Sp, psi_set):
    tol = 0.001
    G_loss = matrix_loss(dets, Sf, Sp)
    G_constraint = np.tile(psi_set[Sp], [len(Sf), 1])
    G = G_loss + G_constraint
    f, v1 = solve_f(G)
    try:
        p, v2 = solve_p(G)
    except:
        print(G)
    assert abs(v1 - v2) < tol
    return f, p, v1
    
def nash_equilibrium(imgpath, theta, dets, psi_set):
    Sp = []
    Sf = []
    #arbitrary pick at first, argmax according to the original paper
    first_y_id = np.argmax(psi_set)
    Sp.append(first_y_id)
    Sf.append(first_y_id)
    lenSp = len(Sp)
    lenSf = len(Sf)
    while True:
        f, p, vp = solveGame(dets, np.array(Sf), np.array(Sp), psi_set) 
        #find y_new
        max_expected_loss = 0.
        max_y_id = -1
        for new_y_id in range(len(psi_set)):
            expected_loss = 0.
            for i, f_id in enumerate(Sf):
                expected_loss += f[i] * (iou_loss1(dets[f_id], dets[new_y_id]) + psi_set[new_y_id])
            if expected_loss > max_expected_loss:
                max_expected_loss = expected_loss
                max_y_id = new_y_id
        vmax = max_expected_loss 
        #if abs(vp - vmax) > tol:
        if max_y_id not in Sp:
            Sp.append(max_y_id)
        f, p, vf = solveGame(dets, np.array(Sf), np.array(Sp), psi_set)
        
        #find y_prime_new
        min_expected_loss = float("inf")
        min_y_prime_id = -1
        for new_y_prime_id in range(len(psi_set)):
            expected_loss = 0.
            for i, p_id in enumerate(Sp):
                expected_loss += p[i] * iou_loss1(dets[p_id], dets[new_y_prime_id])
            if expected_loss < min_expected_loss:
                min_expected_loss = expected_loss
                min_y_prime_id = new_y_prime_id
        vmin = min_expected_loss
        #if abs(vf - vmin) > tol:
        if min_y_prime_id not in Sf:
            Sf.append(min_y_prime_id)
        if lenSp == len(Sp) and lenSf == len(Sf):
        #if abs(vp - vmax) < tol and abs(vmax - vf) < tol and abs(vf - vmin) < tol:
            break
        else:
            lenSp, lenSf = len(Sp), len(Sf)
    return Sf, f, Sp, p

if __name__ == '__main__':
    pass
