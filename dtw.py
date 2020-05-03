import time
import sys
import pandas as pd
import numpy as np

class Warp:
    def __init__(self, x, y, v, p=None):
        self.x = x
        self.y = y
        self.v = v
        self.p = p

    def __lt__(a, b):
        return a.v < b.v

    def __str__(self):
        return "(x: %d, y:%d, v:%.0f)" % (self.x, self.y, self.v)

    def getWarpPath(self):
        r = []
        w = self

        while w.x!=0 or w.y!=0:
            r.insert(0, (w.x, w.y))
            w = w.p

        r.insert(0, (w.x, w.y))
        return r

    def printWarpGrid(self):
        def printGrid(w, n, m):
            if w.x!=0 or w.y!=0:
                printGrid(w.p, n, m)
            else:
                print('x ', end='')
                return
    
            if w.p.y == w.y-1:
                for i in range(w.p.x+1, n):
                    print('- ', end='')
                print()
                for i in range(0, w.x):
                    print('- ', end='')
            
            print('x ', end='')

        printGrid(self, self.x+1, self.y+1)
        print()

class DTW:
    def __init__(self, dfn):
        self.dfn = dfn

    def dist(self, support, query, wp=None, rad=np.Inf):
        n = len(support) + 1
        m = len(query) + 1
        mem = np.empty((n, m), dtype=object)
        twp = []
        for i in np.arange(0, n):
            if wp is None:
                twp.append((i-rad, i+rad))

            for j in np.arange(0, m):
                mem[i, j] = Warp(i-1, j-1, np.Inf)

        mem[0,0].v = 0
        
        if wp is None:
            wp = twp

        for i in range(1, n):
            for j in range(max(1, wp[i-1][0]), min(wp[i-1][1], m)):
                cost = self.dfn(support[i-1], query[j-1])
                w = min(mem[i-1, j], mem[i, j-1], mem[i-1, j-1])
                mem[i, j].v = cost + w.v
                mem[i, j].p = w

        return mem[n - 1, m - 1]
