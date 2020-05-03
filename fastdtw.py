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
        def getPath(w):
            r = []
            if w.x!=0 or w.y!=0:
                r = getPath(w.p)

            r.append((w.x, w.y))
            return r

        return getPath(self)

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

    def dist(self, support, query, wp=None):
        n = len(support) + 1
        m = len(query) + 1
        mem = np.empty((n, m), dtype=object)
        twp = []
        for i in np.arange(0, n):
            if wp is None:
                twp.append((1, m))

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

    def dist(self, support, query, rad=2):
        size = rad+2

        if len(support) <= size or len(query) <= size:
            return self.dtw.dist(support, query)

        print('support', support)
        print('query', query)
        s = support.shrink()
        q = query.shrink()
        print('s', s)
        print('q', q)

        warp = self.dist(s, q, rad)
        print('warp', warp.getWarpPath())
        window = self.searchWindow(warp, s, q)
        print('search window', window)
        window = self.expandWindow(window, rad)
        print('expand window', window)

        return self.dtw.dist(support, query, window)

    def expandWindow(self, win, rad):
        exp = list()
        for w in win:
            mn = min(w)
            mx = max(w)
            exp.append((mn-rad, mx+rad))

        return exp

    def searchWindow(self, warp, support, query):
        wp = warp.getWarpPath()
        sw = list()
        prev_s = None
        for i in range(0, len(wp)):
            w = wp[i]
            for s in support.par[w[0]]:
                if prev_s is not None and prev_s == s:
                    sw[-1].expand(query.par[w[1]])
                else:
                    sw.append(query.par[w[1]].copy())
                    prev_s = s

        return sw

def main():
    s = pd.read_csv("data.csv", names = ["ts", "val"])["val"].tolist()
    q = pd.read_csv("query.csv", names = ["ts", "val"])["val"].tolist()
    n = 10
    m = 10
    query = Timeseries(q[50:50+m])

    d = FastDTW(lambda x, y: abs(x-y))
    #d = DTW(lambda x, y: abs(x-y))

    for i in np.arange(0, len(s)-len(q), 1024):
        support = Timeseries(s[i:i+n])
        w = d.dist(support, query)

        print(w.getWarpPath())
        w.printWarpGrid()
        print(w.v)
        break
        

main()
