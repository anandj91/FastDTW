import time
import sys
import pandas as pd
import numpy as np
from dtw import *

class Bin:
    def __init__(self, val, l=None, r=None, d=0):
        self.val = val
        self.depth = d
        self.left = l
        self.right = r

    def __str__(self):
        return str((self.depth, self.val))

class BinHeap:
    def __init__(self, lim, size=-1):
        self.lim = lim
        self.size = size
        self.lst = []
        self.buf = []

    def insert(self, b):
        if b.depth == self.lim:
            if len(self.lst) == self.size:
                self.lst.pop(0)
            self.lst.append(b)
        else:
            if len(self.buf) > 0 and self.buf[-1].depth == b.depth:
                bl = self.buf.pop()
                nb = Bin((bl.val+b.val)/2, bl, b, d=bl.depth+1)
                self.insert(nb)
            else:
                self.buf.append(b)

    def getLevel(self):
        return self.lst[0].depth if len(self.lst) > 0 else self.lim

    def __len__(self):
        return len(self.lst) + len(self.buf)

    def __getitem__(self, i):
        return self.lst[i] if (i<len(self.lst)) else self.buf[i-len(self.lst)]
    
    def __str__(self):
        r = [str(self[i]) for i in range(len(self))]
        return str(r)

    def par(self, i):
        if i < len(self.lst):
            return [2*i, 2*i+1]
        else:
            return [len(self.lst)+i]

    def lower(self):
        lst = []
        buf = []
        lvl = self.getLevel()
        for l in self.lst:
            lst.append(l.left)
            lst.append(l.right)

        for b in self.buf:
            if b.depth == lvl-1:
                lst.append(b)
            else:
                buf.append(b)

        bh = BinHeap(lvl-1, -1)
        bh.lst = lst
        bh.buf = buf
        return bh

class HTimeseries:
    def __init__(self, h):
        self.heap = h

    def lower(self):
        return HTimeseries(self.heap.lower())

    def __len__(self):
        return len(self.heap)

    def __getitem__(self, i):
        return self.heap[i].val

    def __str__(self):
        return str(self.heap)

    def getLevel(self, i=None):
        if i is not None:
            return self.heap[i].depth
        else:
            return self.heap.getLevel()

    def par(self, i):
        return self.heap.par(i)

class FastDTWStream:
    def __init__(self, dfn):
        self.dfn = dfn
        self.dtw = DTW(dfn)

    def dist(self, support, query, rad=3, depth=0):
        s = support
        q = query

        assert s.getLevel() == q.getLevel()

        warp = self.dtw.dist(s, q)
        while s.getLevel() != depth or q.getLevel() != depth:
            window = self.searchWindow(warp, s, q)
            window = self.expandWindow(window, rad)
            s = s.lower()
            q = q.lower()
            warp = self.dtw.dist(s, q, window)

        return warp

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
            for s in support.par(i):
                sw.append([])
                for j in js:
                    sw[-1].extend(query.par(j))

        return sw

def main():
    s = pd.read_csv("data.csv", names = ["ts", "val"])["val"].tolist()
    q = pd.read_csv("query.csv", names = ["ts", "val"])["val"].tolist()
    lim = int(sys.argv[1])
    size = int(sys.argv[2])
    depth = int(sys.argv[3])

    support = BinHeap(lim, size)
    query = BinHeap(lim, size)

    for i in range(len(q)):
        query.insert(Bin(q[i]))

    fd = FastDTWStream(lambda x, y: abs(x-y))
    step = 1
    for i in range(0, len(s), step):
        for j in range(i, i+step):
            support.insert(Bin(s[i]))
        w_fd = fd.dist(HTimeseries(support), HTimeseries(query), rad=3, depth=depth)
        print("%s, %s" % (s[i], w_fd.v))

if __name__ == "__main__":
    main()
