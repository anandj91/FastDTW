from fastdtw import *
from fastdtw_stream import *
from dtw import *

def main():
    s = pd.read_csv("data.csv", names = ["ts", "val"])["val"].tolist()
    q = pd.read_csv("query.csv", names = ["ts", "val"])["val"].tolist()
    s_off = int(sys.argv[1])
    s_lim = int(sys.argv[2])
    q_off = int(sys.argv[3])
    q_lim = int(sys.argv[4])
    depth = int(sys.argv[5])

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

    support = BinHeap(depth)
    query = BinHeap(depth)
    for i in range(s_off, s_off+s_lim):
        support.insert(Bin(s[i]))
    for i in range(q_off, q_off+q_lim):
        query.insert(Bin(q[i]))

    fd = FastDTWStream(lambda x, y: abs(x-y))
    #d = DTW(lambda x, y: abs(x-y))

    t_fd = time.time()
    w_fd = fd.dist(HTimeseries(support), HTimeseries(query), rad=3)
    print("FastDTW Stream = %s (%s)" % (w_fd.v, time.time()-t_fd))

    #print(w.getWarpPath())
    #w.printWarpGrid()

if __name__ == "__main__":
    main()
