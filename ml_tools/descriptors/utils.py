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
    return slices,strides





