import unittest, numpy as np
from ssg_core import signed_dihedral,triangle_descriptor,triangle_similarity,Node,hierarchical_match,estimate_pose_pnp
class TestSSG(unittest.TestCase):
 def test_dihedral_zero(self):
  d,s=signed_dihedral([0,0,1],[0,0,1],[1,0,0]); self.assertAlmostEqual(d,0); self.assertEqual(s,0)
 def test_triangle_scale_invariance(self):
  p=np.array([[0,0,0],[2,0,0],[.4,1,0]],float)
  self.assertTrue(np.allclose(triangle_descriptor(p),triangle_descriptor(p*5)))
 def test_similarity_identity(self):
  p=np.array([[0,0,0],[2,0,0],[.4,1,0]],float); d=triangle_descriptor(p); self.assertAlmostEqual(triangle_similarity(d,d),1.)
 def test_empty_match(self): self.assertEqual(hierarchical_match([],[]),[])
 def test_pnp_rejects_too_few(self):
  with self.assertRaises(ValueError): estimate_pose_pnp([],[],[],np.eye(3))
if __name__=='__main__': unittest.main()
