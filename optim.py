# optimization tools

import os
import sys
import numpy as np
from time import sleep
from collections import OrderedDict
from multiprocessing import Pool, Process, Queue, Pipe

##
## simple annealing
##

# this always jumps to the better point, so it's like zero temperature annealing
def anneal(f, x0, scale=0.15, decr=0.5, tol=1e-3, wait=50, maxiter=sys.maxsize):
    n = len(x0)
    xp = x0
    yp = f(x0)
    xmin = xp
    ymin = yp
    stay = 0
    for i in range(maxiter):
        xp = xmin*np.exp(scale*np.random.randn(n))
        yp = f(xp)
        if yp < ymin and ~np.isnan(yp):
            xmin = xp
            ymin = yp
            print(f'MIN -> {ymin}: {xmin}')
            stay = 0
        else:
            stay += 1

        if stay == wait:
            scale *= decr
            print(f'SCALE -> {scale}')
            stay = 0

        if scale < tol:
            break

    return xmin, ymin

##
## parallel annealing
##

# parameters tracker
class Tracker():
    def __init__(self, x, y=-np.inf, scale=0.15, decr=0.5, tol=1e-3, wait=50, maxiter=sys.maxsize):
        self.nx = len(x)
        self.scale = scale
        self.decr = decr
        self.tol = tol
        self.wait = wait
        self.maxiter = maxiter
        self.best_x = x
        self.best_y = y
        self.tries = 0
        self.output(x, y)
        print(f'SCALE: {self.scale}')

    def output(self, x, y, i=0):
        xstr = ','.join(['{:8.5f}']*self.nx).format(*x)
        print(f'{i} -> [{xstr}]: {y:15.5g} <= {self.best_y:15.5g} ({self.tries})')

    def random(self):
        return self.best_x + self.scale*np.random.randn(self.nx)

    def update(self, x, y, i=0):
        self.tries += 1
        self.output(x, y, i)

        if y < self.best_y:
            self.best_x = x
            self.best_y = y
            self.tries = 0
        elif self.tries >= self.wait:
            self.scale *= self.decr
            self.tries = 0
            print(f'SCALE: {self.scale}')

        if self.scale <= self.tol or self.tries >= self.maxiter:
            print(f'CONVERGED')
            return True
        else:
            return False

# basic communicating worker
def gen_worker(f):
    def worker(qc):
        while True:
            x = qc.recv()
            y = f(x)
            qc.send((x, y))
    return worker

# generate processes
def startup(f, N):
    for i in range(N):
        qp, qc = Pipe()
        p = Process(target=gen_worker(f), args=(qc,))
        p.start()
        yield p, qp, qc

# anneal entry point
def anneal_parallel(f, x0, N=5, tick=0.1, **kwargs):
    # handle dicts
    x_dict = type(x0) in (dict, OrderedDict)
    if x_dict:
        names = list(x0)
        x1 = np.array([x0[n] for n in names])
        f1 = lambda x: f({n: z for n, z in zip(names, x)})
    else:
        x1 = x0
        f1 = f

    # initialize state
    y1 = f1(x1)
    track = Tracker(x1, y1, **kwargs)
    procs = list(startup(f1, N))

    # initialize workers
    for p, qp, qc in procs:
        qp.send(track.random())

    # loop forever
    done = False
    while True:
        for i, (p, qp, qc) in enumerate(procs):
            if qp.poll():
                x, y = qp.recv()
                done = track.update(x, y, i+1)
                xp = track.random()
                qp.send(xp)
        if done:
            for p, _, _ in procs:
                p.terminate()
            break
        sleep(tick)

    if x_dict:
        best_x = {n: z for n, z in zip(names, track.best_x)}
    else:
        best_x = track.best_x

    # return final results
    return best_x, track.best_y
