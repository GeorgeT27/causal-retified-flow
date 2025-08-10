from typing import Dict

import numpy as np
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
import torch
import torch.nn.functional as F
from layer import (  # fmt: skip
    CNN,
    ConditionalAffineTransform,
)
from pyro.distributions.conditional import ConditionalTransformedDistribution
from pyro.infer.reparam.transform import TransformReparam
from pyro.nn import DenseNN
from torch import Tensor, nn

from hps import Hparams

class BasePGM(nn.Module):
    def __init__(self):
        super().__init__()

    def scm(self, *args, **kwargs):
        def config(msg):
            if isinstance(msg["fn"], dist.TransformedDistribution):
                return TransformReparam()
            else:
                return None

        return pyro.poutine.reparam(self.model, config=config)(*args, **kwargs)

    def sample_scm(self, n_samples: int = 1):
        with pyro.plate("obs", n_samples):
            samples = self.scm()
        return samples

    def sample(self, n_samples: int = 1):
        with pyro.plate("obs", n_samples):
            samples = self.model()  # NOTE: not ideal as model is defined in child class
        return samples

    def infer_exogeneous(self, obs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        batch_size = list(obs.values())[0].shape[0]
        # assuming that we use transformed distributions for everything:
        cond_model = pyro.condition(self.sample, data=obs)
        cond_trace = pyro.poutine.trace(cond_model).get_trace(batch_size)

        output = {}
        for name, node in cond_trace.nodes.items():
            if "z" in name or "fn" not in node.keys():
                continue
            fn = node["fn"]
            if isinstance(fn, dist.Independent):
                fn = fn.base_dist
            if isinstance(fn, dist.TransformedDistribution):
                # compute exogenous base dist (created with TransformReparam) at all sites
                output[name + "_base"] = T.ComposeTransform(fn.transforms).inv(
                    node["value"]
                )
        return output

    def counterfactual(
        self,
        obs: Dict[str, Tensor],
        intervention: Dict[str, Tensor],
        num_particles: int = 1,
        detach: bool = True,
    ) -> Dict[str, Tensor]:
        # NOTE: not ideal as "variables" is defined in child class
        dag_variables = self.variables.keys()
        assert set(obs.keys()) == set(dag_variables)
        avg_cfs = {k: torch.zeros_like(obs[k]) for k in obs.keys()}
        batch_size = list(obs.values())[0].shape[0]

        for _ in range(num_particles):
            # Abduction
            exo_noise = self.infer_exogeneous(obs)
            exo_noise = {k: v.detach() if detach else v for k, v in exo_noise.items()}
            # condition on root node variables (no exogeneous noise available)
            for k in dag_variables:
                if k not in intervention.keys():
                    if k not in [i.split("_base")[0] for i in exo_noise.keys()]:
                        exo_noise[k] = obs[k]
            # Abducted SCM
            abducted_scm = pyro.poutine.condition(self.sample_scm, data=exo_noise)
            # Action
            counterfactual_scm = pyro.poutine.do(abducted_scm, data=intervention)
            # Prediction
            counterfactuals = counterfactual_scm(batch_size)

            for k, v in counterfactuals.items():
                avg_cfs[k] += v / num_particles
        return avg_cfs

class MorphoMNISTPGM(BasePGM):
    def __init__(self, args):
        super().__init__()
        self.variables = {
            "thickness": "continuous",
            "intensity": "continuous",
            "digit": "categorical",
        }
        # priors
        self.digit_logits = nn.Parameter(torch.zeros(1, 10))  # uniform prior
        for k in ["t", "i"]:  # thickness, intensity, standard Gaussian
            self.register_buffer(f"{k}_base_loc", torch.zeros(1))
            self.register_buffer(f"{k}_base_scale", torch.ones(1))

        # constraint, assumes data is [-1,1] normalized
        normalize_transform = T.ComposeTransform(
            [T.SigmoidTransform(), T.AffineTransform(loc=-1, scale=2)]
        )

        # thickness flow
        self.thickness_module = T.ComposeTransformModule(
            [T.Spline(1, count_bins=4, order="linear")]
        )
        self.thickness_flow = T.ComposeTransform(
            [self.thickness_module, normalize_transform]
        )

        # intensity (conditional) flow: thickness -> intensity
        intensity_net = DenseNN(1, args.widths, [1, 1], nonlinearity=nn.GELU())
        self.context_nn = ConditionalAffineTransform(
            context_nn=intensity_net, event_dim=0
        )
        self.intensity_flow = [self.context_nn, normalize_transform]

        if args.setup != "sup_pgm":
            # anticausal predictors
            input_shape = (args.input_channels, args.input_res, args.input_res)
            # q(t | x, i) = Normal(mu(x, i), sigma(x, i)), 2 outputs: loc & scale
            self.encoder_t = CNN(input_shape, num_outputs=2, context_dim=1, width=8)
            # q(i | x) = Normal(mu(x), sigma(x))
            self.encoder_i = CNN(input_shape, num_outputs=2, width=8)
            # q(y | x) = Categorical(pi(x))
            self.encoder_y = CNN(input_shape, num_outputs=10, width=8)
            self.f = (
                lambda x: args.std_fixed * torch.ones_like(x)
                if args.std_fixed > 0
                else F.softplus(x)
            )

    def model(self) -> Dict[str, Tensor]:
        pyro.module("MorphoMNISTPGM", self)
        # p(y), digit label prior dist
        py = dist.OneHotCategorical(
            probs=F.softmax(self.digit_logits, dim=-1)
        )  # .to_event(1)
        # with pyro.poutine.scale(scale=0.05):
        digit = pyro.sample("digit", py)

        # p(t), thickness flow
        pt_base = dist.Normal(self.t_base_loc, self.t_base_scale).to_event(1)
        pt = dist.TransformedDistribution(pt_base, self.thickness_flow)
        thickness = pyro.sample("thickness", pt)

        # p(i | t), intensity conditional flow
        pi_t_base = dist.Normal(self.i_base_loc, self.i_base_scale).to_event(1)
        pi_t = ConditionalTransformedDistribution(
            pi_t_base, self.intensity_flow
        ).condition(thickness)
        intensity = pyro.sample("intensity", pi_t)
        _ = self.context_nn

        return {"thickness": thickness, "intensity": intensity, "digit": digit}

    def guide(self, **obs) -> None:
        # guide for (optional) semi-supervised learning
        with pyro.plate("observations", obs["x"].shape[0]):
            # q(i | x)
            if obs["intensity"] is None:
                i_loc, i_logscale = self.encoder_i(obs["x"]).chunk(2, dim=-1)
                qi_t = dist.Normal(torch.tanh(i_loc), self.f(i_logscale)).to_event(1)
                obs["intensity"] = pyro.sample("intensity", qi_t)

            # q(t | x, i)
            if obs["thickness"] is None:
                t_loc, t_logscale = self.encoder_t(obs["x"], y=obs["intensity"]).chunk(
                    2, dim=-1
                )
                qt_x = dist.Normal(torch.tanh(t_loc), self.f(t_logscale)).to_event(1)
                obs["thickness"] = pyro.sample("thickness", qt_x)

            # q(y | x)
            if obs["digit"] is None:
                y_prob = F.softmax(self.encoder_y(obs["x"]), dim=-1)
                qy_x = dist.OneHotCategorical(probs=y_prob)  # .to_event(1)
                pyro.sample("digit", qy_x)

    def model_anticausal(self, **obs) -> None:
        # assumes all variables are observed & continuous ones are in [-1,1]
        pyro.module("MorphoMNISTPGM", self)
        with pyro.plate("observations", obs["x"].shape[0]):
            # q(t | x, i)
            t_loc, t_logscale = self.encoder_t(obs["x"], y=obs["intensity"]).chunk(
                2, dim=-1
            )
            qt_x = dist.Normal(torch.tanh(t_loc), self.f(t_logscale)).to_event(1)
            pyro.sample("thickness_aux", qt_x, obs=obs["thickness"])

            # q(i | x)
            i_loc, i_logscale = self.encoder_i(obs["x"]).chunk(2, dim=-1)
            qi_t = dist.Normal(torch.tanh(i_loc), self.f(i_logscale)).to_event(1)
            pyro.sample("intensity_aux", qi_t, obs=obs["intensity"])

            # q(y | x)
            y_prob = F.softmax(self.encoder_y(obs["x"]), dim=-1)
            qy_x = dist.OneHotCategorical(probs=y_prob)  # .to_event(1)
            pyro.sample("digit_aux", qy_x, obs=obs["digit"])

    def predict(self, **obs) -> Dict[str, Tensor]:
        # q(t | x, i)
        t_loc, t_logscale = self.encoder_t(obs["x"], y=obs["intensity"]).chunk(
            2, dim=-1
        )
        t_loc = torch.tanh(t_loc)
        # q(i | x)
        i_loc, i_logscale = self.encoder_i(obs["x"]).chunk(2, dim=-1)
        i_loc = torch.tanh(i_loc)
        # q(y | x)
        y_prob = F.softmax(self.encoder_y(obs["x"]), dim=-1)
        return {"thickness": t_loc, "intensity": i_loc, "digit": y_prob}

    def svi_model(self, **obs) -> None:
        with pyro.plate("observations", obs["x"].shape[0]):
            pyro.condition(self.model, data=obs)()

    def guide_pass(self, **obs) -> None:
        pass
