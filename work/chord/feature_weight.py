import numpy as np
def EarthMover(a, b):
    a = np.sort(a) * 1. / sum(a)
    b = np.sort(b) * 1. / sum(b)
    ret = 0.
    i = 0
    j = 0
    u1 = 1.0 / len(a)
    u2 = 1.0 / len(b)
    r1 = u1
    r2 = u2
    while i < len(a) and j < len(b):
        if r1 <= r2:
            ret += r1 * abs(a[i] - b[j])
            r2 -= r1
            r1 = u1
            i += 1
        else:
            ret += r2 * abs(a[i] - b[j])
            r1 -= r2
            r2 = u2
            j += 1
    # print i, r1, j, r2
    return ret

def DistributionWeight(features):
    '''
    (dataset * samples * feature_dimension)
    '''
    print features[0].shape
    max_dis = [0] * features[0].shape[1]
    for i in range(features[0].shape[1]):
        max_dis[i] = 0
        for j in range(len(features)):
            for k in range(j + 1, len(features)):
                # dis = EarthMover(features[j][:, i], features[k][:, i])
                # dis = abs(np.mean(features[j][:, i]) - np.mean(features[k][:, i])) / abs(np.mean(features[j][:, i]))
                p1 = sum(features[j][:, i] < 1e-3) * 1.0 / len(features[j][:, i])
                p2 = sum(features[k][:, i] < 1e-3) * 1.0 / len(features[k][:, i])
                dis = abs(p1 - p2) 
                if(max_dis[i] < dis):
                    max_dis[i] = dis
    # print max_dis
    return max_dis
