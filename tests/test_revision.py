import unittest
import numpy as np
import torch
from experiments.optim import polar_numpy
from experiments.vision import preprocess,peak_rss_mib
from experiments.vision_model import ResNet18,ConvMuon,vision_optimizers
from experiments.factorization import objective


class RevisionTests(unittest.TestCase):
    def test_factorization_two_block_gradient(self):
        rng = np.random.default_rng(7)
        u,v,target = rng.normal(size=(8,4)),rng.normal(size=(6,4)),rng.normal(size=(8,6))
        du,dv = rng.normal(size=u.shape),rng.normal(size=v.shape)
        _,gu,gv = objective(u,v,target)
        eps = 1e-6
        numerical = (objective(u+eps*du,v+eps*dv,target)[0]-objective(u-eps*du,v-eps*dv,target)[0])/(2*eps)
        analytical = float(np.sum(gu*du)+np.sum(gv*dv))
        self.assertAlmostEqual(numerical,analytical,places=6)
    def test_complete_resnet18_and_parameter_partition(self):
        model = ResNet18()
        self.assertEqual(sum(p.numel() for p in model.parameters()),11181642)
        self.assertEqual([len(model.layer1),len(model.layer2),len(model.layer3),len(model.layer4)],[2,2,2,2])
        self.assertEqual(model.conv1.kernel_size,(7,7))
        self.assertEqual(model.conv1.stride,(2,2))
        opts,names = vision_optimizers(model,'muon',0.02)
        ids = [id(p) for opt,_ in opts for g in opt.param_groups for p in g['params']]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(set(ids),{id(p) for p in model.parameters()})
        self.assertNotIn('conv1.weight',names['muon'])
        self.assertNotIn('fc.weight',names['muon'])
        self.assertEqual(tuple(model.eval()(torch.zeros(2,3,32,32)).shape),(2,10))

    def test_k_zero_is_normalized_momentum_and_preserves_grad(self):
        p = torch.nn.Parameter(torch.zeros(4,3,3,3))
        p.grad = torch.arange(p.numel(),dtype=torch.float32).reshape_as(p)
        original = p.grad.clone()
        opt = ConvMuon([p],lr=0.02,steps=0,weight_decay=0)
        opt.step()
        direction = original*.0975
        expected = -0.02*direction/(direction.norm()+1e-7)
        torch.testing.assert_close(p,expected)
        torch.testing.assert_close(p.grad,original)
        for k in [0,1,3,5,7]:
            zero = torch.nn.Parameter(torch.zeros(2,2,3,3))
            zero.grad = torch.zeros_like(zero)
            ConvMuon([zero],steps=k).step()
            self.assertTrue(torch.isfinite(zero).all())
            self.assertEqual(float(zero.detach().abs().sum()),0)

    def test_spectral_error_descent_bound(self):
        rng = np.random.default_rng(28)
        for m,n in [(8,4),(4,8),(5,5)]:
            for delta in [0,.1,.8]:
                w = rng.normal(size=(m,n))
                error = rng.normal(size=(m,n))*.1
                q = polar_numpy(w+error)
                perturbation = rng.normal(size=(m,n))
                perturbation *= delta/np.linalg.norm(perturbation,2) if delta else 0
                output = q+perturbation
                eta = .03
                next_value = .5*np.linalg.norm(w-eta*output)**2
                bound = (.5*np.linalg.norm(w)**2 -eta*(1-delta)*np.linalg.svd(w,compute_uv=False).sum()
                         +2*eta*np.sqrt(min(m,n))*np.linalg.norm(error)
                         +.5*eta**2*min(m,n)*(1+delta)**2)
                self.assertLessEqual(next_value,bound+1e-10)

    def test_augmentation_repeatability_and_memory(self):
        images = torch.arange(4*3*32*32).reshape(4,3,32,32).to(torch.uint8)
        a = preprocess(images,torch.Generator().manual_seed(4))
        b = preprocess(images,torch.Generator().manual_seed(4))
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(tuple(a.shape),(4,3,32,32))
        self.assertGreater(peak_rss_mib(),0)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
