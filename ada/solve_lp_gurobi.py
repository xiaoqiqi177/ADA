#!/usr/bin/python3.6

from gurobipy import *
import numpy as np

def solve_f(G):
    # Create a new model
    m = Model("solvef")
    m.setParam('OutputFlag', False)
    # Create variables
    f = []
    lenf = G.shape[0]
    lenp = G.shape[1]
    for i in range(lenf):
        f.append(m.addVar(vtype=GRB.CONTINUOUS, lb = 0., name="f{}".format(i)))
    v = m.addVar(vtype=GRB.CONTINUOUS, lb = -GRB.INFINITY, name="v")
    # Set objective
    m.setObjective(v, GRB.MINIMIZE)
    m.addConstr(quicksum(f[j] for j in range(lenf)) == 1., name="sumf")
    for i in range(lenp):
        m.addConstr(quicksum(f[j] * G[j, i] for j in range(lenf)) <= v, name="extre{}".format(i))
    m.optimize()
    values_f = np.array([ varx.x for varx in m.getVars()[:-1] ])
    return values_f, m.getVars()[-1].x

def solve_p(G):
    # Create a new model
    m = Model("solvep")
    m.setParam('OutputFlag', False)
    # Create variables
    p = []
    lenf = G.shape[0]
    lenp = G.shape[1]
    for i in range(lenp):
        p.append(m.addVar(vtype=GRB.CONTINUOUS, lb = 0., name="p{}".format(i)))
    v = m.addVar(vtype=GRB.CONTINUOUS, lb = -GRB.INFINITY, name="v")
    # Set objective
    m.setObjective(v, GRB.MAXIMIZE)
    m.addConstr(quicksum(p[j] for j in range(lenp)) == 1., name="sump")
    for i in range(lenf):
        m.addConstr(quicksum(G[i, j] * p[j] for j in range(lenp)) >= v, name="extre{}".format(i))
    m.optimize()
    values_p = np.array([ varx.x for varx in m.getVars()[:-1] ])
    return values_p, m.getVars()[-1].x

if __name__ == '__main__':
    #G = np.array([[0, 0, 0],
    #    [0, 0, 0],
    #    [1, 1, -1]], dtype='float32')
    G = np.array([[-1.]], dtype='float32')
    f = solve_f(G)
    p = solve_p(G)
    print(f)
    print(p)
