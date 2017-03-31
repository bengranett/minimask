import minimask.healpix_projection as hp

def test():

    H = hp.HealpixProjector(nside=2)

    for n in [0, 100]:

        x, y = H.random_sample(1, n=n)

        assert(len(x) == n)
        assert(len(y) == n)

