"""Small meaningful checks for scientific denominators and information boundaries."""
import unittest
import numpy as np
from train_models import metrics, resample, split_plan, normal_response
from map_contract import Evidence, Judgment, candidate_update

class Contracts(unittest.TestCase):
    def test_abstention_does_not_become_correct_block(self):
        m=metrics([1,0,0],[.9,.5,.1],.2,.8)
        self.assertEqual(m['true_block_n'],1)
        self.assertEqual(m['blocked_recall'],.5)
        self.assertAlmostEqual(m['coverage'],2/3)
    def test_prefix_ignores_future(self):
        a=np.arange(100).reshape(50,2).astype(float); b=a.copy(); b[25:]=1000
        np.testing.assert_equal(resample(a,times=np.arange(50)*.1,duration=2.4),resample(b,times=np.arange(50)*.1,duration=2.4))
    def test_map_scope_and_shadow(self):
        g=[[2,2]]; e=Evidence('x',Judgment.PASSABLE)
        self.assertEqual(candidate_update(g,(0,0),e),g)
        self.assertEqual(candidate_update(g,(0,0),e,True),[[0,2]])
    def test_group_split_excludes_unlabelled_holdout(self):
        eps=[dict(family=f,y=y,is_control=False) for f,y in [('P1',1),('P1',-1),('P2',1),('N1',0),('N2',0),('N3',0)]]
        for _,tr,_,te in split_plan(eps):
            self.assertFalse({eps[i]['family'] for i in tr}&{eps[i]['family'] for i in te})
    def test_response_fit_excludes_holdout_and_warmup(self):
        rng=np.random.default_rng(42)
        raw=[rng.normal(size=(20,4)).astype('float32') for _ in range(2)]
        times=[np.arange(20)*.1]*2
        names=['q/right_elbow_joint','qdot/right_elbow_joint','q_cmd/right_elbow_joint','effort/right_elbow_joint']
        eps=[dict(is_control=True,y=1,trial_id='train'),dict(is_control=False,y=-1,trial_id='heldout')]
        a,_,fit=normal_response(raw,times,[0],names,eps)
        changed=[raw[0].copy(),raw[1]+100]
        b,_,fit2=normal_response(changed,times,[0],names,eps)
        np.testing.assert_equal(fit['w'],fit2['w'])
        np.testing.assert_equal(a[0],b[0])
        np.testing.assert_equal(a[0][:4,4:],0)

if __name__=='__main__': unittest.main()
