from sklearn.model_selection._split import KFold as _KFold
from sklearn.model_selection._split import ShuffleSplit as _ShuffleSplit
from sklearn.model_selection._split import (_BaseKFold,
        BaseCrossValidator,_validate_shuffle_split,BaseShuffleSplit)
from sklearn.utils.validation import _num_samples
from sklearn.utils import check_random_state
from abc import ABCMeta, abstractmethod
from sklearn.externals.six import with_metaclass
import collections
from ..base import np,sp

 

class KFold(_KFold):
    def __init__(self, n_splits=3, shuffle=False,random_state=None):
        super(KFold, self).__init__(n_splits, shuffle, random_state)
    def get_params(self):
        params = dict(n_splits=self.n_splits,shuffle=self.shuffle,random_state=self.random_state)
        return params

class ShuffleSplit(_ShuffleSplit):
    def __init__(self,n_splits=10, test_size="default", train_size=None,random_state=None):
        super(ShuffleSplit, self).__init__(n_splits, test_size,train_size, random_state)
    def get_params(self):
        params = dict(n_splits=self.n_splits,test_size=self.test_size,
                    train_size=self.train_size,random_state=self.random_state)
        return params

class EnvironmentalKFold(_BaseKFold):
    def __init__(self, n_splits=3, shuffle=False,random_state=None,mapping=None):
        if mapping is None:
            raise ValueError('a mapping should be provided')
        super(EnvironmentalKFold, self).__init__(n_splits, shuffle, random_state)
        self.mapping = mapping
        try:
            self.mapping[0]
            self.proper_dict = True
        except:
            self.proper_dict = False
            
    def id_map(self,ids):
        if self.proper_dict is True:
            return ids
        elif self.proper_dict is False:
            return str(ids)

    def get_params(self):
        params = dict(n_splits=self.n_splits,shuffle=self.shuffle,
                    random_state=self.random_state,mapping=self.mapping)
        return params
    def _iter_test_indices(self, X, y=None, groups=None):
        #frameid2environmentid = self.mapping
        n_samples = _num_samples(self.mapping)
        indices = np.arange(n_samples)
        if self.shuffle:
            check_random_state(self.random_state).shuffle(indices)

        n_splits = self.n_splits
        fold_sizes = (n_samples // n_splits) * np.ones(n_splits, dtype=np.int)
        fold_sizes[:n_samples % n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            ids = []
            for it in indices[start:stop]:
                ids.extend(self.mapping[self.id_map(it)])
            yield ids
            current = stop

class EnvironmentalShuffleSplit(BaseShuffleSplit):
    def __init__(self, n_splits=3, shuffle=True,random_state=10,mapping=None,train_size=None,test_size="default"):
        if mapping is None or train_size is None:
            raise ValueError('a mapping or number of training frames should be provided')
        super(EnvironmentalShuffleSplit, self).__init__(n_splits,test_size,train_size,random_state)
        self.mapping = mapping
    def get_params(self):
        params = dict(n_splits=self.n_splits,test_size=self.test_size,train_size=self.train_size,
                        random_state=self.random_state,mapping=self.mapping)
        return params
    def _iter_indices(self, X, y=None, groups=None):
        n_samples = _num_samples(self.mapping)
        n_train, n_test = _validate_shuffle_split(n_samples,self.test_size,self.train_size)
        rng = check_random_state(self.random_state)
        for _ in range(self.n_splits):
            # random partition
            permutation = rng.permutation(n_samples)
            ind_test = []
            for it in permutation[:n_test]:
                ind_test.extend(self.mapping[it])
            ind_train = []
            for it in permutation[n_test:(n_test + n_train)]:
                ind_train.extend(self.mapping[it])
            yield ind_train, ind_test

class LCSplit(with_metaclass(ABCMeta)):
    def __init__(self, cv, n_repeats=[10],train_sizes=[10],test_size="default", random_state=None, **cvargs):
        if not isinstance(n_repeats, collections.Iterable) or not isinstance(train_sizes, collections.Iterable):
            raise ValueError("Number of repetitions or training set sizes must be an iterable.")

        if len(n_repeats) != len(train_sizes) :
            raise ValueError("Number of repetitions must be equal to length of training set sizes.")

        if any(key in cvargs for key in ('random_state', 'shuffle')):
            raise ValueError("cvargs must not contain random_state or shuffle.")

        self.cv = cv
        self.n_repeats = n_repeats
        self.train_sizes = train_sizes
        self.random_state = random_state
        self.cvargs = cvargs
        self.test_size = test_size
        self.n_splits = np.sum(n_repeats)
    
    def get_params(self):
        params = dict(cv=self.cv.get_params(),n_repeats=self.n_repeats,train_sizes=self.train_sizes,
                     test_size=self.test_size,random_state=self.random_state,cvargs=self.cvargs)
        return params

    def split(self, X, y=None, groups=None):
        """Generates indices to split data into training and test set.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, of length n_samples
            The target variable for supervised learning problems.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Returns
        -------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """

        rng = check_random_state(self.random_state)
        
        for n_repeat,train_size in zip(self.n_repeats,self.train_sizes):
            cv = self.cv(random_state=rng, n_splits=n_repeat,test_size=self.test_size,train_size=train_size,
                             **self.cvargs)
            for train_index, test_index in cv.split(X, y, groups):
                yield train_index, test_index
                
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator
        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
            ``np.zeros(n_samples)`` may be used as a placeholder.
        y : object
            Always ignored, exists for compatibility.
            ``np.zeros(n_samples)`` may be used as a placeholder.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        rng = check_random_state(self.random_state)
        n_splits = 0
        for n_repeat,train_size in zip(self.n_repeats,self.train_sizes):
            cv = self.cv(random_state=rng, n_splits=n_repeat,test_size=self.test_size,train_size=train_size,
                             **self.cvargs)
            n_splits += cv.get_n_splits(X, y, groups)
        return n_splits
    
