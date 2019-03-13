from ..base import np


def get_chunks(lenght,chunk_lenght):
    Nchunk = lenght // chunk_lenght
    rest = lenght % chunk_lenght
    strides = [s*chunk_lenght for s in range(Nchunk+1)]
    if rest > 0:
        Nchunk += 1
        strides.append(strides[-1]+rest)
    bounds = [(s,e) for s,e in zip(strides[:-1],strides[1:])]
    return bounds

def get_frame_slices(frames,nocenters=None,fast_avg=False):
    if fast_avg:
        slices = []
        strides = [0]
        for frame in frames:
            strides.append(1)
        strides = np.cumsum(strides)
        for st,nd in zip(strides[:-1],strides[1:]):
            slices.append(slice(st,nd))
    else:
        slices = []
        strides = [0]
        for frame in frames:
            numbers = frame.get_atomic_numbers()
            if nocenters is not None:
                numbers = [z for z in numbers if z not in nocenters]
            strides.append(len(numbers))
        strides = np.cumsum(strides)
        for st,nd in zip(strides[:-1],strides[1:]):
            slices.append(slice(st,nd))
    Ncenter = strides[-1]
    return Ncenter,slices,strides


def get_frame_neigbourlist(frames,nocenters):
    Nneighbour = 0
    strides_neighbour = []
    strides_gradients = [0]
    for frame in frames:
        # include centers too wit +1
        numbers = frame.get_atomic_numbers()
        n_neighb = frame.get_array('n_neighb')+1
        mask = np.zeros(numbers.shape,dtype=bool)

        for sp in nocenters:
            mask = np.logical_or(mask, numbers == sp)
        mask = np.logical_not(mask)

        n_neighb = n_neighb[mask]
        Nneighbour += np.sum(n_neighb)
        strides_neighbour += list(n_neighb)
        strides_gradients += [np.sum(n_neighb)]

    strides_gradients = np.cumsum(strides_gradients)
    slices_gradients = []
    for st,nd in zip(strides_gradients[:-1],strides_gradients[1:]):
        slices_gradients.append(slice(st,nd))

    strides_gradients = [0]+strides_neighbour*3

    strides_gradients = np.cumsum(strides_gradients)

    return Nneighbour,strides_gradients,slices_gradients


