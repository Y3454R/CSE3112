# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 12:02:30 2022

@author: ASUS
"""

import numpy as np
def ft(x1, x2, x3):
    return x1*x1 + x2*x2 + x3*x3
particles = 30
vmax = 3
vmin = -3
X = np.random.uniform(-10,10,(particles,3))
V = np.random.uniform(vmin,vmax,(particles,3))
p_best = X
p_best_fit = ft(X[:,0], X[:,1],X[:,2])
g_best = X[np.argmin(p_best_fit)]
g_best_fit = ft(g_best[0], g_best[1], g_best[2])
#parameters
w = 0.9
c1 = 2
c2 = 2
def up():
    global X, V, w, c1, c2, p_best, p_best_fit, g_best, g_best_fit, vmin, vmax
    for i in range(particles):
        r1 = np.random.rand()
        r2 = np.random.rand()
        V = w*V + c1*r1*(p_best-X)+c2*r2*(g_best-X)
        for vel in range(particles):
            for dim in range(3):
                if(V[i,dim] > vmax):
                    V[i,dim] = vmax
                elif(V[i,dim]<vmin):
                    V[i,dim] = vmin
        X = X + V
        new_fit = ft(X[:,0], X[:,1], X[:,2])
        for p in range(particles):
            if(new_fit[p] < p_best_fit[i]):
                p_best_fit[p] = new_fit[p]
                p_best[p] = X[p]
            if(p_best_fit[p] < g_best_fit):
                g_best_fit = p_best_fit[p]
                g_best = p_best[p]

#function
limit = 200
for i in range(limit):
    up()
print(f"g_best_fit = {g_best_fit}\ng_best= {g_best}")