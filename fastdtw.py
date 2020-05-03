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

def main():
    s = pd.read_csv("data.csv", names = ["ts", "val"])["val"].tolist()
    q = pd.read_csv("query.csv", names = ["ts", "val"])["val"].tolist()
    s_off = int(sys.argv[1])
    s_lim = int(sys.argv[2])
    q_off = int(sys.argv[3])
    q_lim = int(sys.argv[4])
    support = Timeseries(s[s_off:s_off+s_lim])
    query = Timeseries(q[q_off:q_off+q_lim])

    fd = FastDTW(lambda x, y: abs(x-y))
    d = DTW(lambda x, y: abs(x-y))

    t_d = time.time()
    w_d = d.dist(support, query)
    print("DTW = %s (%s)" % (w_d.v, time.time()-t_d))

    t_dr = time.time()
    w_dr = d.dist(support, query, rad=3)
    print("DTW with radius = %s (%s)" % (w_dr.v, time.time()-t_dr))

    t_fd = time.time()
    w_fd = fd.dist(support, query, rad=3)
    print("FastDTW = %s (%s)" % (w_fd.v, time.time()-t_fd))

    #print(w.getWarpPath())
    #w.printWarpGrid()

if __name__ == "__main__":
    main()
