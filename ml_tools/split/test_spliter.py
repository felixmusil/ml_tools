from .spliter import EnvironmentalKFold,EnvironmentalSplit,NormalSplit,LCSplit

import unittest
import numpy as np

class TestEnvironmentalKFold(unittest.TestCase):
    def setUp(self):
        """

        """
        self.trains = [np.array([5, 6, 7]), np.array([0, 1, 2, 3, 4, 7]), np.array([0, 1, 2, 3, 4, 5, 6])]
        self.tests = [np.array([0, 1, 2, 3, 4]), np.array([5, 6]), np.array([7])]
        self.mapping = {0:[0,1],1:[2,3,4],2:[5,6],3:[7,]}

    def test_EnvironmentalKFold(self):
        """
        """

        cv = EnvironmentalKFold(n_splits=3, shuffle=False,random_state=None,mapping=self.mapping)
        trains = []
        tests = []
        for train,test in cv.split(np.ones((8,))):
            trains.append(train)
            tests.append(test)
        for a,b,c,d in zip(trains,tests,self.trains,self.tests):
            self.assertTrue(np.allclose(a,c))
            self.assertTrue(np.allclose(b,d))