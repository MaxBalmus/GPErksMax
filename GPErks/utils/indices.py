import numpy as np
from time import time

def diff(l1, l2):
    return list(set(l1) - set(l2))


def inters(l1, l2):
    return list(set(l1) & set(l2))


def inters_many(L):
    return list(set.intersection(*map(lambda x: set(x), L)))


def union_many(L):
    return list(set.union(*map(lambda x: set(x), L)))


def restrict_kth_comp(data, k, ib, ub):
    l = []
    for i in range(data.shape[0]):
        if (
            np.where(data[i, k] > ib)[0].shape[0]
            and np.where(data[i, k] < ub)[0].shape[0]
        ):
            l.append(i)
    return l


def find_start_seq(index, feat_dim):  # utility function for "whereq_whernot"
    i = 0
    while i < len(index):
        if index[i : feat_dim + i] == list(range(feat_dim)):
            return i
        else:
            i += 1
    return


def delta(X):  # utility function for "part_and_select"
    # compute per‐column min and max in one go
    col_min = X.min(axis=0)
    col_max = X.max(axis=0)
    ranges = col_max - col_min

    # pick the index and its range
    pj = int(np.argmax(ranges))
    return pj, float(ranges[pj])


def part_and_select_1(P, N):
    C1 = P.copy()
    min_ = C1.min(axis=0)
    max_ = C1.max(axis=0)
    C1 = (C1 - min_) / (max_ - min_)
    p1, s1 = delta(C1)
    archive = [(s1, p1, C1)]
    i = 1
    print('Part and select')
    print('Step 1 start')
    time0 = time()
    while i < N:
        if all([not x[0] for x in archive]):
            break
        sj, pj, Cj = max(archive[:i], key=lambda x: x[0])
        cpj = (np.max(Cj[:, pj]) + np.min(Cj[:, pj])) / 2
        Cj1 = Cj[Cj[:,pj] <= cpj]
        Cj2 = Cj[Cj[:,pj] >  cpj]
        pj1, sj1 = delta(Cj1)
        pj2, sj2 = delta(Cj2)
        archive.remove((sj, pj, Cj))
        archive.append((sj1, pj1, Cj1))
        archive.append((sj2, pj2, Cj2))
        i += 1
    selected = []
    time1 = time()
    print(f'Step 1 end {time1 - time0:.2f} s.')
    print('Step 2 start')
    for _, _, C in archive:
        c = np.mean(C, axis=0)
        idx = np.argmin(np.linalg.norm(C - c, axis=1))
        selected.append(C[idx])
    time2 = time()
    print(f'Step 2 end {time2 - time1:.2f} s.')
    return np.stack(selected) * (max_ - min_) + min_

def part_and_select(P, N):
    C1 = P.copy()
    min_ = C1.min(axis=0)
    max_ = C1.max(axis=0)
    C1 = (C1 - min_) / (max_ - min_)
    p1, s1 = delta(C1)
    n1 = C1.shape[0]
    archive = np.zeros((2*N-1,3))
    flags   = np.full((2*N-1,), True)
    range_  = np.arange(2*N-1)
    archive[0,:]  = s1, p1, n1
    archive_array = np.zeros((n1, 2*N+1), dtype=np.uint32)
    archive_array[:n1,0] = np.arange(n1, dtype=np.uint32)
    k, j = 0, 1
    print('Part and select')
    print('Step 1')
    print('Step 1 start')
    time0 = time()
    time00 = 0.
    for i in range(1,N):
        time01 = time()
        subset = archive[flags][:i]  # slice once
        ind = int(np.argmax(subset[:, 0]))
        time00 += time() - time01
        sj, pj, nj = subset[ind]
        if sj < 1e-16: break
        pj, nj = int(pj), int(nj)
        ind2 = range_[flags][ind]
        flags[ind2] = False
        Cj_ind = archive_array[:nj,ind2]
        cpj = (np.max(C1[Cj_ind, pj]) + np.min(C1[Cj_ind, pj])) / 2
        Cj_flags = (C1[Cj_ind,pj] <= cpj)
        nj1 = Cj_flags.sum()
        nj2 = nj - nj1
        archive_array[:nj1,j]   = Cj_ind[Cj_flags]
        archive_array[:nj2,j+1] = Cj_ind[~Cj_flags]
        pj1, sj1 = delta(C1[archive_array[:nj1,j]])
        pj2, sj2 = delta(C1[archive_array[:nj2,j+1]])
        archive[j ] [0:3] = np.array([sj1, pj1, nj1])
        archive[j+1][0:3] = np.array([sj2, pj2, nj2])
        j += 2
        i += 1
        k += 1
    time1 = time()
    print(flags.shape)
    print(f'Step 1 end {time1 - time0:.2f} s., search time {time00:.2f} s.')
    print('Step 2')
    print('Step 2 start')
    selected = []
    for i in range_[flags[:j]]:
        nj = archive[i,2]
        C = C1[archive_array[:int(nj),i]]
        c = np.mean(C, axis=0)
        idx = np.argmin(np.linalg.norm(C - c, axis=1))
        selected.append(C[idx])
    time2 = time()
    print(f'Step 2 end {time2 - time1:.2f} s.')
    return np.stack(selected) * (max_ - min_) + min_

def matrix_subtraction(X, XS):
    set_X = set(map(tuple, X))
    set_XS = set(map(tuple, XS))
    return np.array(list(set_X - set_XS))


def whereq_whernot(X, SX):
    feat_dim = X.shape[1]
    l = []
    for i in range(SX.shape[0]):
        index = np.where(X == SX[i, :])
        if len(list(index[1])) > feat_dim:
            l.append(index[0][find_start_seq(list(index[1]), feat_dim)])
        else:
            l.append(index[0][0])
    nl = diff(range(X.shape[0]), l)
    nl.sort()
    return l, nl


def filter_zscore(X, thre):
    feat_dim = X.shape[1]
    L = []
    for j in range(feat_dim):
        z = np.abs((X[:, j] - np.mean(X[:, j])) / np.std(X[:, j]))
        L.append(np.where(z > thre)[0])
    nl = union_many(L)
    l = diff(range(X.shape[0]), nl)
    return l, nl
