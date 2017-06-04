import numpy as np

def vote(names):
    ret = None
    for name in names:
        y = np.load(name + '.npy')
        if(ret is None):
            ret =  y
        else:
            ret = ret + y
    return ret
np.set_printoptions(precision=4, suppress=True)
names1 = ['dm_bird', 'rf100_bird', 'svm_bird', './luogan/bird', 'gdbt_bird']
ans1 = vote(names1)
rank = range(len(ans1))
rank = sorted(rank, key = lambda x: -ans1[x])
np.save('vote_bird.npy',np.array(ans1 <= ans1[rank[110]]))
names2 = ['dm_cat', 'rf100_cat', 'svmpoly_cat', './luogan/cat', 'gdbt_cat']
ans2 = vote(names2)
rank2 = range(len(ans2))
rank2 = sorted(rank2, key = lambda x: -ans2[x])
np.save('vote_cat.npy', np.array(ans2 <= ans2[rank2[130]]))