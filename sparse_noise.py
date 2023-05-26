import numpy as np


def create_spots(

    vadims=(120, 90),
    nits=100,
    spotsize=5,
    ndots=12,
    nlim=20
):

    sa = np.zeros(dtype=np.uint8, shape=(int(round(vadims[0]/spotsize)), int(round(vadims[1]/spotsize)))) + 255/2
    npix = sa.ravel().shape[0]
    sat = np.array([sa]*npix*2, dtype=np.uint8)
    sat[np.arange(0, npix, 1),np.where(sa)[0],np.where(sa)[1]] = 255
    sat[npix+np.arange(0, npix, 1),np.where(sa)[0],np.where(sa)[1]] = 0
    spotc = np.zeros(shape=(2, sa.shape[0], sa.shape[1]), dtype=np.uint16)
    nims = 0
    ims = []
    for _ in range(nits):

        ids = np.random.randint(0, sat.shape[0], ndots)

        crds = np.array(np.where(sat[ids].mean(0) != np.median(sat[ids]))).T
        crm = np.concatenate([np.roll(crds, i, 0) for i in range(crds.shape[0])], axis=1)
        cdif = (crm[1:] - crm[0]) ** 2
        dists = np.sqrt(cdif[:, ::2] + cdif[:, 1::2])

        if _ % 100 == 0:
            print(_)
        if not np.any(dists < 2):

            satmin, satmax = sat[ids].min(0), sat[ids].max(0)
            satmax[np.where(satmin == 0)] = 0

            if len(np.where(satmax == 0)[0]) > 0:
                spotmax = spotc[0, np.where(satmax == 0)[0], np.where(satmax == 0)[1]].max()
            else:
                spotmax = 0
            if len(np.where(satmax == 255)[0]) > 0:
                spotmin = spotc[1, np.where(satmax == 255)[0], np.where(satmax == 255)[1]].max()
            else:
                spotmin = 0
            if np.any(np.array([spotmin, spotmax]) > nlim):
                continue
            nims += 1
            ims.append(satmax.astype(np.uint8))
            spotc[0, np.where(satmax == 0)[0], np.where(satmax == 0)[1]] += 1
            spotc[1, np.where(satmax == 255)[0], np.where(satmax == 255)[1]] += 1
            # plt.imshow(satmax.T, aspect='equal', cmap='gray')
            # plt.show()

    print(spotc.min(), spotc.max(), np.median(spotc), spotc.mean())
    ims = np.array(ims).astype(np.uint8)
    return ims
if __name__ == "__main__":

    #ims = create_spots(nits=1000000, spotsize=3, ndots=24)
    ims = create_spots(nits=1000000, spotsize=4.65, ndots=16, vadims=(128, 74))

    np.save('sparse_ts_dims128_74_size465_ndots16.npy', ims)