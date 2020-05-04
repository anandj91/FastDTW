import time
import sys
import pandas as pd
import numpy as np
from dtw import *

class Timeseries:
    def __init__(self, lst, par=None):
        self.lst = lst

        if par is None:
            self.par = []
            for i, e in enumerate(self.lst):
                self.par.append(i)
        else:
            self.par = par

    def __len__(self):
        return len(self.lst)

    def __getitem__(self, i):
        return self.lst[i]

    def __str__(self):
        s = "["
        for p, l in zip(self.par, self.lst):
            s += ("(%s, %s)," % (p, l))
        s += "]"
        return s

    def shrink(self):
        l = self.lst
        lst = list()
        par = list()

        for i in np.arange(1, len(l), 2):
            lst.append((l[i-1]+l[i])/2)
            par.append([i-1, i])
        
        if len(l) % 2 != 0:
            lst.append(l[-1])
            par.append([len(l)-1])

        return Timeseries(lst, par)

class FastDTW:
    def __init__(self, dfn):
        self.dfn = dfn
        self.dtw = DTW(dfn)

    def dist(self, support, query, rad=3):
        size = rad+2

        if len(support) <= size or len(query) <= size:
            return self.dtw.dist(support, query)

        s = support.shrink()
        q = query.shrink()

        warp = self.dist(s, q, rad)
        window = self.searchWindow(warp, s, q)
        window = self.expandWindow(window, rad)

        return self.dtw.dist(support, query, window)

    def expandWindow(self, win, rad):
        exp = list()
        for w in win:
            mn = min(w)
            mx = max(w)
            exp.append((mn-rad, mx+rad))

        return exp

    def searchWindow(self, warp, support, query):
        wp = []
        prev_s = None
        for w in warp.getWarpPath():
            if prev_s is not None and prev_s == w[0]:
                wp[-1].append(w[1])
            else:
                wp.append([w[1]])
                prev_s = w[0]

        sw = list()
        for i in range(0, len(wp)):
            js = wp[i]
            for s in support.par[i]:
                sw.append([])
                for j in js:
                    sw[-1].extend(query.par[j].copy())

        return sw
