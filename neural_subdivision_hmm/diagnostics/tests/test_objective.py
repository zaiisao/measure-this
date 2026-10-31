import math
import unittest

import torch

from neural_subdivision_hmm.diagnostics.objective import aggregate_terms,objective_from_emissions,sequence_terms
from neural_subdivision_hmm.model import NeuralGaussianEmission
from subdivision_hmm.hmm import forward_log_prob,persistence_log_transition


class NllDecompositionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.model=NeuralGaussianEmission(m=8,hidden=12,rank=1)
        self.init=torch.full((5,),-math.log(5.0))
        self.trans=persistence_log_transition(5,.75)
        self.y=torch.randn(6,8)
        self.d=torch.linspace(-.2,.2,6)
        self.emission=self.model.log_prob(self.y,self.d)

    def test_joint_equals_acoustic_plus_posterior_state_nll(self):
        allowed=torch.ones(6,5,dtype=torch.bool)
        allowed[1]=False; allowed[1,1]=True
        allowed[4]=False; allowed[4,2]=True; allowed[4,4]=True
        terms=sequence_terms(self.init,self.trans,self.emission,allowed)
        self.assertTrue(torch.allclose(terms.joint_nll,terms.acoustic_nll+terms.state_nll,atol=1e-6))
        expected=-forward_log_prob(self.init,self.trans,self.emission,allowed)
        self.assertTrue(torch.allclose(terms.joint_nll,expected,atol=1e-6))

    def test_all_unknown_state_nll_is_exactly_zero(self):
        allowed=torch.ones(6,5,dtype=torch.bool)
        terms=sequence_terms(self.init,self.trans,self.emission,allowed)
        self.assertEqual(float(terms.state_nll),0.0)
        loss=aggregate_terms([terms],lambda_state=10)
        self.assertEqual(float(loss.state_nll_per_informative),0.0)
        self.assertTrue(torch.allclose(loss.loss,loss.acoustic_nll_per_observed))

    def test_posterior_state_term_has_finite_nonzero_emission_gradient(self):
        allowed=torch.ones(6,5,dtype=torch.bool)
        allowed[:]=False; allowed[:,1]=True
        terms=sequence_terms(self.init,self.trans,self.emission,allowed)
        terms.state_nll.backward()
        grads=[p.grad for p in self.model.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(float(g.abs().sum()) for g in grads),0)

    def test_constraints_at_missing_audio_keep_state_and_transition(self):
        allowed=torch.ones(6,5,dtype=torch.bool)
        allowed[2]=False; allowed[2,3]=True
        observed=torch.tensor([True,True,False,True,True,True])
        changed=self.emission.clone(); changed[2]=1000
        a=sequence_terms(self.init,self.trans,self.emission,allowed,observed)
        b=sequence_terms(self.init,self.trans,changed,allowed,observed)
        self.assertTrue(torch.allclose(a.acoustic_nll,b.acoustic_nll,atol=1e-6))
        self.assertTrue(torch.allclose(a.joint_nll,b.joint_nll,atol=1e-6))
        # Removing a missing row is not equivalent because it removes a transition;
        # neutralizing its emission in the six-row chain is the intended operation.
        neutral=torch.where(observed[:,None],self.emission,torch.zeros_like(self.emission))
        expected=-forward_log_prob(self.init,self.trans,neutral,allowed)
        self.assertTrue(torch.allclose(a.joint_nll,expected,atol=1e-6))

    def test_lambda_one_equals_joint_nll_per_observed_interval(self):
        masks=[]
        for n,positions in [(3,[0]),(5,[1,2,4])]:
            mask=torch.ones(n,5,dtype=torch.bool)
            for t in positions:
                mask[t]=False; mask[t,(t+1)%5]=True
            masks.append(mask)
        emissions=[self.emission[:3],self.emission[1:6]]
        observed=[torch.tensor([True,True,False]),torch.ones(5,dtype=torch.bool)]
        terms=objective_from_emissions(self.init,self.trans,emissions,masks,observed,lambda_state=1)
        self.assertEqual(terms.observed_intervals,7)
        self.assertEqual(terms.informative_intervals,4)
        self.assertTrue(torch.allclose(terms.loss,terms.joint_nll_sum/7,atol=1e-6))
        self.assertTrue(torch.allclose(terms.state_nll_per_informative,terms.state_nll_sum/4,atol=1e-6))

    def test_lambda_scales_only_posterior_supervision_component(self):
        allowed=torch.ones(6,5,dtype=torch.bool)
        allowed[0]=False; allowed[0,0]=True
        obs=torch.ones(6,dtype=torch.bool)
        one=objective_from_emissions(self.init,self.trans,[self.emission],[allowed],[obs],1)
        five=objective_from_emissions(self.init,self.trans,[self.emission],[allowed],[obs],5)
        self.assertTrue(torch.allclose(five.loss-one.loss,4*one.state_nll_sum/6,atol=1e-6))


if __name__=='__main__': unittest.main()
