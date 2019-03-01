
from ..base import KernelBase,FeatureBase
from ..base import np,sp
from ..math_utils import power,average_kernel
from ..utils import tqdm_cs,is_autograd_instance
#from ..math_utils.basic import power,average_kernel
from scipy.sparse import issparse
# from collections.abc import Iterable


class KernelPower(KernelBase):
    def __init__(self,zeta):
        self.zeta = zeta
    def fit(self,X):
        return self
    def get_params(self,deep=True):
        params = dict(zeta=self.zeta)
        return params
    def set_params(self,**params):
        self.zeta = params['zeta']
    def transform(self,X,X_train=None):
        return self(X,Y=X_train)

    def __call__(self, X, Y=None, eval_gradient=False):
        # X should be shape=(Nsample,Mfeature)
        # if not assumes additional dims are features
        if isinstance(X, FeatureBase):
            Xi = X.representations
        elif len(X.shape) > 2:
            Nenv = X.shape[0]
            Xi = X.reshape((Nenv,-1))
        else:
            Xi = X

        if isinstance(Y, FeatureBase):
            Yi = Y.representations
        elif Y is None:
            Yi = Xi
        elif len(Y.shape) > 2:
            Nenv = Y.shape[0]
            Yi = Y.reshape((Nenv,-1))
        else:
            Yi = Y


        if issparse(Xi) is False:
            return power(np.dot(Xi,Yi.T),self.zeta)
        if issparse(Xi) is True:
            N, M = Xi.shape[0],Yi.shape[0]
            kk = np.zeros((N,M))
            Xi.dot(Yi.T).todense(out=kk)
            return power(kk,self.zeta)

    def pack(self):
        state = dict(zeta=self.zeta)
        return state
    def unpack(self,state):
        err_m = 'zetas are not consistent {} != {}'.format(self.zeta,state['zeta'])
        assert self.zeta == state['zeta'], err_m

    def loads(self,state):
        self.zeta = state['zeta']


class KernelSum(KernelBase):
    def __init__(self,kernel,chunk_shape=None,disable_pbar=False):
        self.kernel = kernel
        self.chunk_shape = chunk_shape
        self.disable_pbar = disable_pbar
    def fit(self,X):
        return self
    def get_params(self,deep=True):
        params = dict(kernel=self.kernel,disable_pbar=self.disable_pbar,
                    chunk_shape=self.chunk_shape)
        return params
    def set_params(self,**params):
        self.kernel = params['kernel']
        self.chunk_shape = params['chunk_shape']
        self.disable_pbar = params['disable_pbar']

    def __call__(self, X, Y=None):
        return self.transform(X, Y)

    def transform(self,X,X_train=None, eval_gradient=False):
        if isinstance(X,FeatureBase):
            Xfeat,Xstrides = X.get_data(eval_gradient)
        elif isinstance(X,dict):
            Xfeat,Xstrides = X['feature_matrix'], X['strides']

        is_square = False
        if X_train is None:
            Yfeat,Ystrides = Xfeat,Xstrides
            is_square = True
        elif isinstance(X_train,FeatureBase):
            Yfeat,Ystrides = X_train.get_data(gradients=False)
        elif isinstance(X_train,dict):
            Yfeat,Ystrides = X_train['feature_matrix'], X_train['strides']

        if self.chunk_shape is None:
            K = self.get_global_kernel(Xfeat,Xstrides,Yfeat,Ystrides,is_square)
        elif is_autograd_instance(Xfeat) is True or is_autograd_instance(Yfeat) is True:
            # this function is autograd safe because it avoids
            K = self.get_global_kernel_chunk_autograd(Xfeat,Xstrides,Yfeat,Ystrides,is_square)
        else:
            K = self.get_global_kernel_chunk(Xfeat,Xstrides,Yfeat,Ystrides,is_square)
        return K

    def get_global_kernel(self,Xfeat,Xstrides,Yfeat,Ystrides,is_square):
        envKernel = self.kernel(Xfeat,Yfeat)
        K = average_kernel(envKernel,Xstrides,Ystrides,is_square)
        return K

    def get_global_kernel_chunk(self,Xfeat,Xstrides,Yfeat,Ystrides,is_square_kernel):
        chunk_shape = self.chunk_shape
        Nframe1,Nframe2 = len(Xstrides)-1,len(Ystrides)-1
        kernel = np.ones((Nframe1,Nframe2))
        if chunk_shape is None:
            chunk_shape = [Nframe1,Nframe2]
        if chunk_shape[0] > Nframe1:
            chunk_shape[0] = Nframe1
        if chunk_shape[1] > Nframe2:
            chunk_shape[1] = Nframe2
        if is_square_kernel is True:
            chunk_shape[1] = chunk_shape[0]

        chunks1,chunks2 = get_chunck(Xstrides,chunk_shape[0]),get_chunck(Ystrides,chunk_shape[1])
        total = 0
        for it,(fids1,ch1,ch1g) in enumerate(zip(*chunks1)):
            #if Nchunk_max < ch1[-1].stop - ch1[0].start: Nchunk_max = ch1[-1].stop - ch1[0].start
            for jt,(fids2,ch2,ch2g) in enumerate(zip(*chunks2)):
                #if Mchunk_max < ch2[-1].stop - ch2[0].start: Mchunk_max = ch2[-1].stop - ch2[0].start
                if is_square_kernel:
                    if jt >= it: total += 1
                    else: continue
                else: total += 1

        pbar = tqdm_cs(total=total,desc='kernel chunks',leave=False,disable=self.disable_pbar)
        kk = []
        for it,(fids1,ch1,ch1g) in enumerate(zip(*chunks1)):
            for jt,(fids2,ch2,ch2g) in enumerate(zip(*chunks2)):
                is_square = False
                if is_square_kernel:
                    if jt > it: continue

                sl1,sl2 = slice(ch1g[0][0],ch1g[-1][1]), slice(ch2g[0][0],ch2g[-1][1])
                ss1,ss2 = Xfeat[sl1],Yfeat[sl2]

                Xstrides_local = [st for st,nd in ch1]+[ch1[-1][1]]
                Ystrides_local = [st for st,nd in ch2]+[ch2[-1][1]]
                envKernel = self.kernel(ss1,ss2)

                kernels_chunk = average_kernel(envKernel,Xstrides_local,
                                            Ystrides_local,is_square=is_square)
                #print kernels_chunk.shape
                kernel[fids1,fids2] = kernels_chunk
                if is_square_kernel:
                    if get_overlap(fids1,fids2) == 0:
                        kernel[fids2,fids1] = kernel[fids1,fids2].T
                pbar.update()
        pbar.close()
        return kernel

    def get_global_kernel_chunk_autograd(self,Xfeat,Xstrides,Yfeat,Ystrides,is_square_kernel):
        chunk_shape = self.chunk_shape
        Nframe1,Nframe2 = len(Xstrides)-1,len(Ystrides)-1
        kernel = np.ones((Nframe1,Nframe2))
        if chunk_shape is None:
            chunk_shape = [Nframe1,Nframe2]
        if chunk_shape[0] > Nframe1:
            chunk_shape[0] = Nframe1
        if chunk_shape[1] > Nframe2:
            chunk_shape[1] = Nframe2
        if is_square_kernel is True:
            chunk_shape[1] = chunk_shape[0]

        chunks1,chunks2 = get_chunck(Xstrides,chunk_shape[0]),get_chunck(Ystrides,chunk_shape[1])
        Nchunk, Mchunk = len(chunks1[0]),len(chunks2[0])
        total = 0
        for it,(fids1,ch1,ch1g) in enumerate(zip(*chunks1)):
            #if Nchunk_max < ch1[-1].stop - ch1[0].start: Nchunk_max = ch1[-1].stop - ch1[0].start
            for jt,(fids2,ch2,ch2g) in enumerate(zip(*chunks2)):
                #if Mchunk_max < ch2[-1].stop - ch2[0].start: Mchunk_max = ch2[-1].stop - ch2[0].start
                if is_square_kernel:
                    if jt > it:
                        total += 1
                        continue
                else: total += 1

        pbar = tqdm_cs(total=total,desc='kernel chunks',leave=False,disable=self.disable_pbar)

        kk = {it:{jt:{} for jt in range(Mchunk)} for it in range(Nchunk)}
        for it,(fids1,ch1,ch1g) in enumerate(zip(*chunks1)):
            for jt,(fids2,ch2,ch2g) in enumerate(zip(*chunks2)):
                is_square = False
                if is_square_kernel is True:
                    if jt > it: continue

                sl1,sl2 = slice(ch1g[0][0],ch1g[-1][1]), slice(ch2g[0][0],ch2g[-1][1])
                ss1,ss2 = Xfeat[sl1],Yfeat[sl2]

                Xstrides_local = [st for st,nd in ch1]+[ch1[-1][1]]
                Ystrides_local = [st for st,nd in ch2]+[ch2[-1][1]]
                envKernel = self.kernel(ss1,ss2)

                kernels_chunk = average_kernel(envKernel,Xstrides_local,
                                            Ystrides_local,is_square=is_square)

                kk[it][jt] = kernels_chunk
                if is_square_kernel is True and get_overlap(fids1,fids2) == 0:
                    kk[jt][it] = kernels_chunk.T
                pbar.update()

        kki = []
        for it in range(Nchunk):
            kki.append(np.concatenate(kk[it].values(),axis=1))
        kernel = np.concatenate(kki,axis=0)

        pbar.close()
        return kernel


    def pack(self):
        state = self.get_params()
        return state
    def unpack(self,state):
        pass

    def loads(self,state):
        self.set_params(state)


class KernelSparseSoR(KernelBase):
    def __init__(self,kernel,X_pseudo,Lambda):
        self.Lambda = Lambda # \sim std of the training properties
        self.kernel = kernel
        self.X_pseudo = X_pseudo

    def fit(self,X):
        return self
    def get_params(self,deep=True):
        params = dict(kernel=self.kernel,X_pseudo=self.X_pseudo,
                    Lambda=self.Lambda)
        return params
    def set_params(self,**params):
        self.kernel = params['kernel']
        self.X_pseudo = params['X_pseudo']
        self.Lambda = params['Lambda']

    def transform(self,X,y=None,X_train=None):
        if X_train is None and y is not None:
            Xs = self.X_pseudo

            kMM = self.kernel.transform(Xs)
            kMN = self.kernel.transform(Xs,X_train=X)
            ## assumes Lambda= Lambda**2*np.diag(np.ones(n))
            sparseK = kMM + np.dot(kMN,kMN.T)/self.Lambda**2
            sparseY = np.dot(kMN,y)/self.Lambda**2

            return sparseK,sparseY
        else:
            return self.kernel.transform(X,X_train=self.X_pseudo)

    def pack(self):
        state = self.get_params()
        return state
    def unpack(self,state):
        pass

    def loads(self,state):
        self.set_params(state)



def get_chunck(strides,chunk_len):
    N = len(strides)-1
    Nchunk = N // chunk_len

    slices = [(st,nd) for st,nd in zip(strides[:-1],strides[1:])]

    if Nchunk == 1 and N % chunk_len == 0:
        return [slice(0,N)],[slices],[slices]

    frame_ids = [slice(it*chunk_len,(it+1)*chunk_len) for it in range(Nchunk)]
    if N % chunk_len != 0:
        frame_ids.append(slice((Nchunk)*chunk_len,(N)))

    chuncks_global = [
        [slices[ii] for ii in range(it*chunk_len,(it+1)*chunk_len)]
              for it in range(Nchunk)]
    if N % chunk_len != 0:
        chuncks_global.append([slices[ii] for ii in range((Nchunk)*chunk_len,(N))])

    chuncks = []
    for it in range(Nchunk):
        st_ref = slices[it*chunk_len][0]
        aa = []
        for ii in range(it*chunk_len,(it+1)*chunk_len):
            st,nd = slices[ii][0]-st_ref,slices[ii][1]-st_ref
            aa.append((st,nd))
        chuncks.append(aa)

    if N % chunk_len != 0:
        aa = []
        st_ref = slices[Nchunk*chunk_len][0]
        for ii in range((Nchunk)*chunk_len,N):
            st,nd = slices[ii][0]-st_ref,slices[ii][1]-st_ref
            aa.append((st,nd))
        chuncks.append(aa)

    return frame_ids,chuncks,chuncks_global

def get_overlap(a, b):
    return max(0, min(a.stop, b.stop) - max(a.start, b.start))
