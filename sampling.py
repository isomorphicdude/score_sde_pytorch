# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""
import functools

import torch
import numpy as np
import abc

from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate
import sde_lib
from models import utils as mutils

_CORRECTORS = {}
_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def register_corrector(cls=None, *, name=None):
    """A decorator for registering corrector classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _CORRECTORS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _CORRECTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]


def get_corrector(name):
    return _CORRECTORS[name]


def get_sampling_fn(config, sde, shape, inverse_scaler, eps):
    """Create a sampling function.

    Args:
      config: A `ml_collections.ConfigDict` object that contains all configuration information.
      sde: A `sde_lib.SDE` object that represents the forward SDE.
      shape: A sequence of integers representing the expected shape of a single sample.
      inverse_scaler: The inverse data normalizer function.
      eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

    Returns:
      A function that takes random states and a replicated training state and outputs samples with the
        trailing dimensions matching `shape`.
    """

    sampler_name = config.sampling.method
    modified = config.sampling.use_preconditioner
    extra_args = config.sampling.extra_args

    # Probability flow ODE sampling with black-box ODE solvers
    if sampler_name.lower() == "ode":
        sampling_fn = get_ode_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            denoise=config.sampling.noise_removal,
            eps=eps,
            device=config.device,
        )
    # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
    elif sampler_name.lower() == "pc":
        predictor = get_predictor(config.sampling.predictor.lower())
        corrector = get_corrector(config.sampling.corrector.lower())

        if config.sampling.use_preconditioner:
            sampling_fn = get_pc_sampler(
                sde=sde,
                shape=shape,
                predictor=predictor,
                corrector=corrector,
                inverse_scaler=inverse_scaler,
                snr=config.sampling.snr,
                n_steps=config.sampling.n_steps_each,
                probability_flow=config.sampling.probability_flow,
                continuous=config.training.continuous,
                denoise=config.sampling.noise_removal,
                eps=eps,
                device=config.device,
                modified_pc=modified,
                extra_args=extra_args,
                return_all=config.eval.make_gif
            )
        else:
            sampling_fn = get_pc_sampler(
                sde=sde,
                shape=shape,
                predictor=predictor,
                corrector=corrector,
                inverse_scaler=inverse_scaler,
                snr=config.sampling.snr,
                n_steps=config.sampling.n_steps_each,
                probability_flow=config.sampling.probability_flow,
                continuous=config.training.continuous,
                denoise=config.sampling.noise_removal,
                eps=eps,
                device=config.device,
                return_all=config.evaluate.make_gif
            )
    else:
        raise ValueError(f"Sampler name {sampler_name} unknown.")

    return sampling_fn


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the predictor.

        Args:
          x: A PyTorch tensor representing the current state
          t: A Pytorch tensor representing the current time step.

        Returns:
          x: A PyTorch tensor of the next state.
          x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the corrector.

        Args:
          x: A PyTorch tensor representing the current state
          t: A PyTorch tensor representing the current time step.

        Returns:
          x: A PyTorch tensor of the next state.
          x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


@register_predictor(name="euler_maruyama")
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t):
        dt = -1.0 / self.rsde.N
        z = torch.randn_like(x)
        drift, diffusion = self.rsde.sde(x, t)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        return x, x_mean


@register_predictor(name="reverse_diffusion")
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False, extra_args=None):
        super().__init__(sde, score_fn, probability_flow)
        if extra_args is not None:
            self.sde_lr = extra_args["sde_lr"]
            self.scale = True

    def update_fn(self, x, t, extra_inputs=None):
        f, G = self.rsde.discretize(x, t)
        z = torch.randn_like(x)
        
        if self.scale:
            x_mean = x - f * self.sde_lr
            x = x_mean + G[:, None, None, None] * z * np.sqrt(self.sde_lr)
                
        else:
            x_mean = x - f
            x = x_mean + G[:, None, None, None] * z
            
        return x, x_mean, None


@register_predictor(name="rms_reverse_diffusion")
class RMSDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False, extra_args=None):
        super().__init__(sde, score_fn, probability_flow)
        if extra_args is None:
            raise ValueError("RMSDiffusionPredictor requires extra arguments.")

        self.beta_pred = extra_args["beta_pred"]
        self.sde_lr = extra_args["sde_lr"]
        self.adam_like = extra_args["adam_like"] # whether to use adam-like update
        self.lamb = extra_args["lamb"] # the extreme values of pre-conditioner
        
        self.interpolation_type = extra_args["interpolation_type"] # either sigmoid or linear
        self.scale = extra_args["scale"]
        self.shift = extra_args["shift"]
        
        self.min_beta = extra_args["min_beta"]
        self.max_beta = extra_args["max_beta"]
        self.debug_mode = extra_args["debug_mode"]


    def update_fn(self, x, t, extra_inputs=None):
        """Returns 3 outputs for update step."""
        
        # the moving average of the squared gradient
        V = extra_inputs["V"]
        counter = extra_inputs["counter"]
        
        d_forward_drift, d_G = self.sde.discretize(x, t)
        score = self.score_fn(x, t)
        d_sub_term = (d_G[:, None, None, None] ** 2) * score
        
        if self.interpolation_type == "sigmoid":
            beta_pred = self.min_beta + (self.max_beta-self.min_beta) * self.sigmoid(counter, 
                                                                                 self.scale, 
                                                                                 self.shift, 
                                                                                 self.sde.N)
        elif self.interpolation_type == "linear":
            beta_pred = self.min_beta + (self.max_beta-self.min_beta) * counter / self.sde.N
        else:
            beta_pred = self.beta_pred

        # update m
        V = beta_pred * V + (1 - beta_pred) * (score**2)

        # construct f with preconditioning
        if self.adam_like:
            f = d_forward_drift - d_sub_term / torch.sqrt(V + self.lamb)
        else:
            # f = d_forward_drift - d_sub_term / (torch.sqrt(V) + self.lamb)
            
            # use clipping instead
            f = d_forward_drift - d_sub_term / (torch.clamp(torch.sqrt(V), min=self.lamb))

        # construct noise with preconditioning, note the double sqrt
        
        if self.adam_like:
            z = torch.randn_like(x) / torch.sqrt(torch.sqrt(V + self.lamb))
        else:
            # z = torch.randn_like(x) / (torch.sqrt(torch.sqrt(V) + self.lamb))
            
            # use clipping instead
            z = torch.randn_like(x) / (torch.sqrt(torch.clamp(torch.sqrt(V), min=self.lamb)))
            
        # else:
        #     if self.adam_like:
        #         z = (
        #             np.sqrt(1 / self.beta4 * 1 / (counter + 1))
        #             * torch.randn_like(x)
        #             / torch.sqrt(torch.sqrt(V + self.lamb))
        #         )
        #     else:
        #         z = (
        #             np.sqrt(1 / self.beta4 * 1 / (counter + 1))
        #             * torch.randn_like(x)
        #             / (torch.sqrt(torch.sqrt(V) + self.lamb))
        #         )

        # update x_mean (no noise at the last step)
        x_mean = x - f * self.sde_lr

        # update x
        x = x_mean + d_G[:, None, None, None] * z * np.sqrt(self.sde_lr)

        
        # x_mean = x + d_G[:, None, None, None] * score * self.sde_lr / (torch.clamp(torch.sqrt(V), min=self.lamb))

        # update counter
        counter += 1
        
        if not self.debug_mode:
            return x, x_mean, {"V": V, "counter": counter}
        else:
            return x, x_mean, {"V": V, 
                               "counter": counter, 
                               "score": score}
    
    def sigmoid(self, x, scale, shift, num_steps):
        """Sigmoid function for interpolation."""
        # here x is the counter indicating the current iteration
        return torch.sigmoid(torch.tensor(scale * (x-shift)/num_steps))
    
    
@register_predictor(name="adam_reverse_diffusion")
class AdamDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False, extra_args=None):
        super().__init__(sde, score_fn, probability_flow)
        if extra_args is None:
            raise ValueError("AdamDiffusionPredictor requires extra arguments.")

        self.beta1 = extra_args["beta1"]
        self.beta2 = extra_args["beta2"]
        self.sde_lr = extra_args["sde_lr"]
        self.correct_bias = extra_args["correct_bias"]
        self.adam_like = extra_args["adam_like"] # whether to use adam-like update
        self.lamb = extra_args["lamb"] # the extreme values of pre-conditioner
        
        self.debug_mode = extra_args["debug_mode"]


    def update_fn(self, x, t, extra_inputs=None):
        """Returns 3 outputs for update step."""
        
        # the moving average of the squared gradient
        m = extra_inputs["m"]
        v = extra_inputs["v"]
        counter = extra_inputs["counter"]
        
        d_forward_drift, d_G = self.sde.discretize(x, t) # the G is multiplied by the sqrt
        score = self.score_fn(x, t)

        # update m and v
        m = self.beta1 * m + (1 - self.beta1) * score
        v = self.beta2 * v + (1 - self.beta2) * (score**2)
        
        # correct for bias
        if self.correct_bias:
            m_hat = m / (1 - self.beta1 ** (counter + 1))
            v_hat = v / (1 - self.beta2 ** (counter + 1))
        else:
            m_hat = m
            v_hat = v
        
        # here the score in sub_term is replaced by first moment
        d_sub_term = (d_G[:, None, None, None] ** 2) * m_hat

        # construct f with preconditioning
        if self.adam_like:
            f = d_forward_drift - d_sub_term / torch.sqrt(v_hat + self.lamb)
        else:
            f = d_forward_drift - d_sub_term / (torch.sqrt(v_hat) + self.lamb)

        # construct noise with preconditioning, note the double sqrt
        if self.adam_like:
            z = torch.randn_like(x) / torch.sqrt(torch.sqrt(v_hat + self.lamb))
        else:
            z = torch.randn_like(x) / (torch.sqrt(torch.sqrt(v_hat) + self.lamb))

        # update x
        x_mean = x - f * self.sde_lr

        # update x_mean (no noise at the last step)
        x = x_mean + d_G[:, None, None, None] * z * np.sqrt(self.sde_lr)

        # update counter
        counter += 1
        
        if not self.debug_mode:
            return x, x_mean, {"m": m, 
                               "v": v,
                               "counter": counter}
        else:
            return x, x_mean, {"m": m, 
                               "v": v,
                               "counter": counter, 
                               "score": score}


@register_predictor(name="ancestral_sampling")
class AncestralSamplingPredictor(Predictor):
    """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)
        if not isinstance(sde, sde_lib.VPSDE) and not isinstance(sde, sde_lib.VESDE):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )
        assert (
            not probability_flow
        ), "Probability flow not supported by ancestral sampling"

    def vesde_update_fn(self, x, t):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        sigma = sde.discrete_sigmas[timestep]
        adjacent_sigma = torch.where(
            timestep == 0,
            torch.zeros_like(t),
            sde.discrete_sigmas.to(t.device)[timestep - 1],
        )
        score = self.score_fn(x, t)
        x_mean = x + score * (sigma**2 - adjacent_sigma**2)[:, None, None, None]
        std = torch.sqrt(
            (adjacent_sigma**2 * (sigma**2 - adjacent_sigma**2)) / (sigma**2)
        )
        noise = torch.randn_like(x)
        x = x_mean + std[:, None, None, None] * noise
        return x, x_mean

    def vpsde_update_fn(self, x, t):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        beta = sde.discrete_betas.to(t.device)[timestep]
        score = self.score_fn(x, t)
        x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1.0 - beta)[
            :, None, None, None
        ]
        noise = torch.randn_like(x)
        x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise
        return x, x_mean

    def update_fn(self, x, t):
        if isinstance(self.sde, sde_lib.VESDE):
            return self.vesde_update_fn(x, t)
        elif isinstance(self.sde, sde_lib.VPSDE):
            return self.vpsde_update_fn(x, t)


@register_predictor(name="none")
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, sde, score_fn, probability_flow=False, extra_args=None):
        pass

    def update_fn(self, x, t, extra_inputs=None):
        return x, x, None


@register_corrector(name="langevin")
class LangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps, extra_args=None):
        super().__init__(sde, score_fn, snr, n_steps)
        if (
            not isinstance(sde, sde_lib.VPSDE)
            and not isinstance(sde, sde_lib.VESDE)
            and not isinstance(sde, sde_lib.subVPSDE)
        ):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )

    def update_fn(self, x, t, extra_inputs=None):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        for i in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

        return x, x_mean, None


@register_corrector(name="rms_langevin")
class RMSLangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps, extra_args=None):
        super().__init__(sde, score_fn, snr, n_steps)
        if (
            not isinstance(sde, sde_lib.VPSDE)
            and not isinstance(sde, sde_lib.VESDE)
            and not isinstance(sde, sde_lib.subVPSDE)
        ):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )

        if extra_args is None:
            raise ValueError("extra_args must be provided for RMSLangevinCorrector")

        # get additional learning rate
        self.lr = extra_args["lr"]

        # parameters
        self.beta1 = extra_args["beta1"]
        self.beta3 = extra_args["beta3"]
        # if self.beta3 > 0:
        #     raise NotImplementedError("beta3 > 0 not yet supported")

    def update_fn(self, x, t, extra_inputs=None):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr

        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        m = extra_inputs["m"]
        counter = extra_inputs["counter"]

        for i in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()

            # check nan
            # print("Debugging RMSLangevinCorrector")
            # print(grad.to('cpu').numpy())
            # if torch.isnan(grad).any():
            #   print(f'iteration {i} in RMSLangevinCorrector')
            #   print(counter)
            #   raise ValueError("grad is nan.")
            # print("End debugging RMSLangevinCorrector")

            # update m
            m = self.beta1 * m + (1 - self.beta1) * (grad**2)

            # adjust step size with extra learning rate, no snr
            step_size = self.lr * ((noise_norm / grad_norm) ** 2) * 2 * alpha

            # update without noise
            x_mean = x + step_size[:, None, None, None] * grad / torch.sqrt(m + 1e-7)

            # update with noise
            if self.beta3 == 0:
                x = x_mean + torch.sqrt(step_size * 2)[
                    :, None, None, None
                ] * noise / torch.sqrt(torch.sqrt(m + 1e-7))
            else:
                x = x_mean + torch.sqrt(step_size * 2)[
                    :, None, None, None
                ] * noise / torch.sqrt(torch.sqrt(m + 1e-7)) * (1 / self.beta3) * (
                    1 / torch.sqrt(counter + 1)
                )

            counter += 1

        return x, x_mean, {"m": m, "counter": counter}


@register_corrector(name="ald")
class AnnealedLangevinDynamics(Corrector):
    """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

    We include this corrector only for completeness. It was not directly used in our paper.
    """

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if (
            not isinstance(sde, sde_lib.VPSDE)
            and not isinstance(sde, sde_lib.VESDE)
            and not isinstance(sde, sde_lib.subVPSDE)
        ):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )

    def update_fn(self, x, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        std = self.sde.marginal_prob(x, t)[1]

        for i in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x)
            step_size = (target_snr * std) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]

        return x, x_mean


@register_corrector(name="rmsald")
class RMSAnnealedLangevinDynamics(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps, extra_args=None):
        super().__init__(sde, score_fn, snr, n_steps)
        if (
            not isinstance(sde, sde_lib.VPSDE)
            and not isinstance(sde, sde_lib.VESDE)
            and not isinstance(sde, sde_lib.subVPSDE)
        ):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )

        if extra_args is None:
            raise ValueError(
                "extra_args must be provided for RMSAnnealedLangevinDynamics"
            )

        # get additional learning rate
        self.lr = extra_args["lr"]

        # ema parameters
        self.beta1 = extra_args["beta1"]
        self.beta3 = extra_args["beta3"]

    def update_fn(self, x, t, extra_inputs=None):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        std = self.sde.marginal_prob(x, t)[1]

        # get ema of squared gradient
        m = extra_inputs["m"]
        counter = extra_inputs["counter"]

        for i in range(n_steps):
            grad = score_fn(x, t)

            if self.beta3 == 0:
                noise = torch.randn_like(x)
            else:
                noise = (1 / self.beta3) * (1 / (counter + 1)) * torch.randn_like(x)

            # update moving average of squared gradient
            m = self.beta3 * m + (1 - self.beta3) * (grad**2)

            # adjusting stepsize for RMSprop
            step_size = ((target_snr * std) ** 2) * 2 * alpha * self.lr

            x_mean = x + step_size[:, None, None, None] * grad / torch.sqrt(m + 1e-7)
            x = x_mean + noise * torch.sqrt(step_size * 2)[
                :, None, None, None
            ] / torch.sqrt(torch.sqrt(m + 1e-7))

            # update counter
            counter += 1

        return x, x_mean, {"m": m, "counter": counter}


@register_corrector(name="none")
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, sde, score_fn, snr, n_steps, extra_args=None):
        pass

    def update_fn(self, x, t, extra_inputs=None):
        return x, x, {}


def shared_predictor_update_fn(
    x,
    t,
    sde,
    model,
    predictor,
    probability_flow,
    continuous,
    modified_pc=False,
    extra_args=None,
    extra_inputs=None,
):
    """A wrapper that configures and returns the update function of predictors."""
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)

    # if modified_pc:
    if predictor is None:
        # Corrector-only sampler
        predictor_obj = NonePredictor(sde, score_fn, probability_flow, extra_args)
    else:
        predictor_obj = predictor(sde, score_fn, probability_flow, extra_args)
    return predictor_obj.update_fn(x, t, extra_inputs)

    # else:
        # # keep original
        # if predictor is None:
        #     # Corrector-only sampler
        #     predictor_obj = NonePredictor(sde, score_fn, probability_flow)
        # else:
        #     predictor_obj = predictor(sde, score_fn, probability_flow)
        # return predictor_obj.update_fn(x, t)


def shared_corrector_update_fn(
    x,
    t,
    sde,
    model,
    corrector,
    continuous,
    snr,
    n_steps,
    modified_pc=False,
    extra_args=None,
    extra_inputs=None,
):
    """A wrapper tha configures and returns the update function of correctors."""
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)

    # if modified_pc:
    if corrector is None:
        # Predictor-only sampler
        corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps, extra_args)
    else:
        corrector_obj = corrector(sde, score_fn, snr, n_steps, extra_args)
    return corrector_obj.update_fn(x, t, extra_inputs)
    # else:
    #     if corrector is None:
    #         # Predictor-only sampler
    #         corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
    #     else:
    #         corrector_obj = corrector(sde, score_fn, snr, n_steps)
    #     return corrector_obj.update_fn(x, t)


def get_pc_sampler(
    sde,
    shape,
    predictor,
    corrector,
    inverse_scaler,
    snr,
    n_steps=1,
    probability_flow=False,
    continuous=False,
    denoise=True,
    eps=1e-3,
    device="cuda",
    modified_pc=False,
    extra_args=None,
    return_all=False,
):
    """Create a Predictor-Corrector (PC) sampler.

    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      device: PyTorch device.
      modified_pc: If `True`, use the user-defined PC sampler.
      extra_args: A dictionary of extra arguments to be passed to initialise the predictor and corrector.
      return_all: If `True`, return all intermediate samples and the number of function evaluations during sampling.

    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    # Create predictor & corrector update functions
    predictor_update_fn = functools.partial(
        shared_predictor_update_fn,
        modified_pc=modified_pc,
        extra_args=extra_args,
        sde=sde,
        predictor=predictor,
        probability_flow=probability_flow,
        continuous=continuous,
    )
    
    corrector_update_fn = functools.partial(
        shared_corrector_update_fn,
        modified_pc=modified_pc,
        extra_args=extra_args,
        sde=sde,
        corrector=corrector,
        continuous=continuous,
        snr=snr,
        n_steps=n_steps,
    )

    
    def pc_sampler(model):
        """The PC sampler funciton.

        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        if return_all:
            all_samples = []  
            
        debug_mode = extra_args["debug_mode"]
        
        if debug_mode:
            #TODO: add other statistics to return for debugging
            score_list = []
            V_list = [] # store moving average
            m_list = []
            
        with torch.no_grad():
            #TODO: Need to change the SDE prior to any input
            #TODO: Need to change the SDE initial time sde.T
            # Initial sample
            
            x = sde.prior_sampling(shape).to(device)
            timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

            
            # initialize the extra inputs
            # if corrector is not None and corrector.__name__ == "RMSLangevinCorrector":
            #     extra_inputs_corr = {"V": torch.zeros_like(x), "counter": 0}
            extra_inputs_corr = None
            extra_inputs_pred = None
            
            if predictor.__name__ == "RMSDiffusionPredictor":
                extra_inputs_pred = {"V": torch.zeros_like(x), 
                                     "counter": 0}
                
            if predictor.__name__ == "AdamDiffusionPredictor":
                extra_inputs_pred = {"m": torch.zeros_like(x), 
                                     "v": torch.zeros_like(x),
                                     "counter": 0}
            # TODO: add other predictors

            for i in range(sde.N):
                t = timesteps[i]
                vec_t = torch.ones(shape[0], device=t.device) * t
                x, x_mean, extra_inputs_corr = corrector_update_fn(
                    x, vec_t, model=model, extra_inputs=extra_inputs_corr
                )
                x, x_mean, extra_inputs_pred = predictor_update_fn(
                    x, vec_t, model=model, extra_inputs=extra_inputs_pred
                )
                
                if return_all:
                    all_samples.append(inverse_scaler(x_mean if denoise else x))
                    
                if debug_mode and predictor.__name__ == "RMSDiffusionPredictor":
                    score_list.append(extra_inputs_pred["score"])
                    V_list.append(extra_inputs_pred["V"])
                    
                elif debug_mode and predictor.__name__ == "AdamDiffusionPredictor":
                    score_list.append(extra_inputs_pred["score"])
                    V_list.append(extra_inputs_pred["v"])
                    m_list.append(extra_inputs_pred["m"])
                        
        if return_all:
            return all_samples, sde.N * (n_steps + 1)
        elif debug_mode:
            return inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1), \
                score_list, V_list, m_list
        else:
            return inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1)

    return pc_sampler



def get_ode_sampler(
    sde,
    shape,
    inverse_scaler,
    denoise=False,
    rtol=1e-5,
    atol=1e-5,
    method="RK45",
    eps=1e-3,
    device="cuda",
):
    """Probability flow ODE sampler with the black-box ODE solver.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      inverse_scaler: The inverse data normalizer.
      denoise: If `True`, add one-step denoising to final samples.
      rtol: A `float` number. The relative tolerance level of the ODE solver.
      atol: A `float` number. The absolute tolerance level of the ODE solver.
      method: A `str`. The algorithm used for the black-box ODE solver.
        See the documentation of `scipy.integrate.solve_ivp`.
      eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
      device: PyTorch device.

    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """

    def denoise_update_fn(model, x):
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        # Reverse diffusion predictor for denoising
        predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
        vec_eps = torch.ones(x.shape[0], device=x.device) * eps
        _, x = predictor_obj.update_fn(x, vec_eps)
        return x

    def drift_fn(model, x, t):
        """Get the drift function of the reverse-time SDE."""
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t)[0]

    def ode_sampler(model, z=None):
        """The probability flow ODE sampler with black-box ODE solver.

        Args:
          model: A score model.
          z: If present, generate samples from latent code `z`.
        Returns:
          samples, number of function evaluations.
        """
        with torch.no_grad():
            # Initial sample
            if z is None:
                # If not represent, sample the latent code from the prior distibution of the SDE.
                x = sde.prior_sampling(shape).to(device)
            else:
                x = z

            def ode_func(t, x):
                x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
                vec_t = torch.ones(shape[0], device=x.device) * t
                drift = drift_fn(model, x, vec_t)
                return to_flattened_numpy(drift)

            # Black-box ODE solver for the probability flow ODE
            solution = integrate.solve_ivp(
                ode_func,
                (sde.T, eps),
                to_flattened_numpy(x),
                rtol=rtol,
                atol=atol,
                method=method,
            )
            nfe = solution.nfev
            x = (
                torch.tensor(solution.y[:, -1])
                .reshape(shape)
                .to(device)
                .type(torch.float32)
            )

            # Denoising is equivalent to running one predictor step without adding noise
            if denoise:
                x = denoise_update_fn(model, x)

            x = inverse_scaler(x)
            return x, nfe

    return ode_sampler
