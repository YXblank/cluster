import unittest
import numpy as np
from core import dihedral, descriptor, similarity, extract
class Test(unittest.TestCase):
    def test_angle(self):
        self.assertAlmostEqual(dihedral([1,0,0],[0,1,0],[0,0,1])[0],np.pi/2)
        self.assertEqual(dihedral([1,0,0],[0,1,0],[0,0,-1])[1],-1)
    def test_triangle(self):
        x=np.array([[0,0,0],[1,0,0],[0,2,0]])
        self.assertAlmostEqual(similarity(descriptor(x),descriptor(3*x+5)),1)
        with self.assertRaises(ValueError): descriptor([[0,0,0],[1,0,0],[2,0,0]])
    def test_extract(self):
        d=np.ones((10,10)); labels=np.ones_like(d,dtype=int); K=np.array([[10,0,5],[0,10,5],[0,0,1]])
        self.assertEqual(len(extract(d,K,labels,labels,labels)),1)
if __name__=='__main__': unittest.main()
