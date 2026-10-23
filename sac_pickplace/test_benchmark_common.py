"""Physics-free correctness tests for the unified success definition."""
import unittest
from benchmark_common import BenchmarkRules, audit_events, audit_final
import numpy as np

class TestAudit(unittest.TestCase):
    def test_sustained_events_success(self):
        rules = BenchmarkRules()
        states = []
        for k in range(40):
            states.append(dict(time_s=.04*(k+1), rise_mm=50.,horizontal_moved_mm=40. if k>=8 else 0.,
                table_contacts=0,left_contacts=1,right_contacts=1))
        self.assertEqual(audit_events(states,rules),(True,True))
        post=[]
        for k in range(50):
            post.append(dict(time_s=2.0+0.002*(k+1),cube_xyz=np.array([.6,.06,.40]), xy_error_mm=2.,
                bottom_gap_mm=0.,speed_m_s=0.,table_contacts=2,left_contacts=0,right_contacts=0))
        score = audit_final(post,True,True,rules)
        self.assertTrue(score['unified_success'])
    def test_contact_spike_not_lift(self):
        rules=BenchmarkRules()
        states=[]
        for k in range(10):
            states.append(dict(time_s=.04*(k+1),rise_mm=50.,horizontal_moved_mm=50.,
                table_contacts=0,left_contacts=1 if k==3 else 0,right_contacts=1 if k==3 else 0))
        self.assertEqual(audit_events(states,rules),(False,False))
    def test_post_settle_drift_failure(self):
        rules=BenchmarkRules()
        post=[]
        for k in range(500):
            post.append(dict(time_s=1+0.002*(k+1),cube_xyz=np.array([.6+.00005*k,.06,.40]),xy_error_mm=2.,
                bottom_gap_mm=0.,speed_m_s=0.,table_contacts=2,left_contacts=0,right_contacts=0))
        score=audit_final(post,True,True,rules)
        self.assertFalse(score['unified_success'])
        self.assertFalse(score['drift_ok'])

if __name__=='__main__':unittest.main()
