import itertools
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data import allowed_states
from model import NeuralGaussianEmission, exclude_missing, forward_backward, forward_log_prob, persistence_log_transition, viterbi


class NeuralHmmTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model=NeuralGaussianEmission(m=8,hidden=12,rank=2)
        self.r=torch.tensor([0.1,-0.2,0.0])
        self.y=torch.randn(3,8)
        self.init=torch.full((5,),-math.log(5.0))
        self.trans=persistence_log_transition(5,.7)

    def test_gaussian_log_density_matches_multivariate_normal(self):
        mu,var,u=self.model.distribution_parameters(self.r)
        logp=self.model.log_prob(self.y,self.r)
        i,k=1,3
        cov=torch.diag(var[i,k])+u[i,k]@u[i,k].T
        ref=torch.distributions.MultivariateNormal(mu[i,k],covariance_matrix=cov).log_prob(self.y[i])
        self.assertTrue(torch.allclose(logp[i,k],ref,atol=2e-4,rtol=2e-4))

    def test_covariance_is_positive_definite(self):
        _,var,u=self.model.distribution_parameters(self.r)
        cov=torch.diag_embed(var)+u@u.transpose(-1,-2)
        self.assertTrue(torch.all(var>0))
        self.assertGreater(float(torch.linalg.eigvalsh(cov).min()),0.0)

    def test_finite_density_and_missing_emission_is_neutral(self):
        lp=self.model.log_prob(self.y,self.r)
        self.assertTrue(torch.isfinite(lp).all())
        observed=torch.tensor([True,False,True])
        neutral=exclude_missing(lp,observed)
        self.assertTrue(torch.equal(neutral[1],torch.zeros(5)))
        masked_z=forward_log_prob(self.init,self.trans,neutral)
        changed=lp.clone(); changed[1]=1000.0
        self.assertTrue(torch.allclose(masked_z,forward_log_prob(self.init,self.trans,exclude_missing(changed,observed)),atol=1e-6))

    def test_forward_likelihood_matches_brute_force(self):
        e=torch.randn(4,5)
        z=forward_log_prob(self.init,self.trans,e)
        terms=[]
        for path in itertools.product(range(5),repeat=4):
            v=self.init[path[0]]+e[0,path[0]]
            for t in range(1,4): v=v+self.trans[path[t-1],path[t]]+e[t,path[t]]
            terms.append(v)
        self.assertTrue(torch.allclose(z,torch.logsumexp(torch.stack(terms),0),atol=1e-5))

    def test_partial_forward_likelihood_matches_brute_force(self):
        e=torch.randn(3,5); allowed=torch.ones(3,5,dtype=torch.bool)
        allowed[0]=False; allowed[0,2]=True
        allowed[2]=False; allowed[2,1]=True; allowed[2,3]=True
        z=forward_log_prob(self.init,self.trans,e,allowed)
        terms=[]
        for path in itertools.product(range(5),repeat=3):
            if all(allowed[t,path[t]] for t in range(3)):
                v=self.init[path[0]]+e[0,path[0]]
                for t in range(1,3): v=v+self.trans[path[t-1],path[t]]+e[t,path[t]]
                terms.append(v)
        self.assertTrue(torch.allclose(z,torch.logsumexp(torch.stack(terms),0),atol=1e-5))

    def test_viterbi_matches_brute_force_map(self):
        e=torch.randn(4,5); path=viterbi(self.init,self.trans,e)
        vals=[]; paths=[]
        for candidate in itertools.product(range(5),repeat=4):
            v=self.init[candidate[0]]+e[0,candidate[0]]
            for t in range(1,4): v=v+self.trans[candidate[t-1],candidate[t]]+e[t,candidate[t]]
            vals.append(v); paths.append(candidate)
        self.assertEqual(tuple(path.tolist()),paths[int(torch.stack(vals).argmax())])

    def test_forward_backward_marginals_normalize(self):
        e=torch.randn(8,5); post,z=forward_backward(self.init,self.trans,e)
        self.assertTrue(torch.allclose(post.sum(-1),torch.ones(8),atol=1e-6))
        self.assertTrue(torch.isfinite(z))

    def test_forward_gradient_is_finite_and_nonzero(self):
        e=self.model.log_prob(self.y,self.r)
        loss=-forward_log_prob(self.init,self.trans,e)
        loss.backward()
        grads=[p.grad for p in self.model.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(float(g.abs().sum()) for g in grads),0.0)

    def test_weak_ambiguous_evidence_training_step_is_stable(self):
        weak=torch.zeros(12,8)
        duration=torch.zeros(12)
        ll=self.model.log_prob(weak,duration)
        loss=-forward_log_prob(self.init,self.trans,ll)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters()))

    def test_marginal_likelihood_dominates_any_path_joint(self):
        e=self.model.log_prob(self.y,self.r)
        z=forward_log_prob(self.init,self.trans,e)
        path=(0,1,1)
        joint=self.init[path[0]]+e[0,path[0]]
        for t in range(1,len(path)): joint=joint+self.trans[path[t-1],path[t]]+e[t,path[t]]
        self.assertGreaterEqual(float(z+1e-5),float(joint))

    def test_unknown_constraints_are_identical_to_unlabeled(self):
        e=self.model.log_prob(self.y,self.r); all_allowed=torch.ones_like(e,dtype=torch.bool)
        self.assertTrue(torch.allclose(forward_log_prob(self.init,self.trans,e),forward_log_prob(self.init,self.trans,e,all_allowed)))

    def test_annotation_constraints_preserve_unknown_and_family_states(self):
        self.assertTrue(allowed_states(0,'exact').all())
        self.assertEqual(np.flatnonzero(allowed_states(8,'family')).tolist(),[1,3])
        self.assertEqual(np.flatnonzero(allowed_states(12,'family')).tolist(),[2,4])

    def test_synthetic_model_emissions_recover_grid_states(self):
        model=NeuralGaussianEmission(m=48,hidden=8,rank=0)
        head=model.generator[-1]
        with torch.no_grad():
            head.weight.zero_(); head.bias.zero_(); head.bias[1]=4.0
            head.bias[2+48:2+96]=-3.0
        T=80; path=[]; state=2
        rng=np.random.default_rng(11)
        for t in range(T):
            if t and rng.random()>.96: state=int(rng.integers(0,5))
            path.append(state)
        labels=torch.tensor(path)
        d=torch.zeros(T)
        mu,var,_=model.distribution_parameters(d)
        y=mu[torch.arange(T),labels]+torch.randn(T,48)*var[torch.arange(T),labels].sqrt()
        ll=model.log_prob(y,d)
        init=torch.full((5,),-math.log(5.0)); trans=persistence_log_transition(5,.7)
        pred=viterbi(init,trans,ll)
        self.assertGreater(float((pred==labels).float().mean()),.85)


if __name__=='__main__': unittest.main()
