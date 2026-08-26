#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gp_uncertainty.py — surrogate for the GP predictive covariance Sigma_GP.

In the thesis, Sigma_GP comes from the trained GP residual of the Hybrid
PEM-GP model (Chapter 2). For a runnable twin campaign this module provides
three modes (config/vstmpc_params.yaml -> gp.mode):

  * "constant": Sigma_GP = sigma2_base * I  (crude but safe)
  * "regime":   variance grows with |attitude| and |angular rates|,
                reproducing the Chapter 2 finding that GP uncertainty grows
                in high-dynamic regimes and shrinks in hover.  (default)
  * "trained":  load your real trained GP (pickle) and query its posterior
                covariance.  Fill `pickle_path` and implement `_query_trained`
                to match your GP library (GPy / GPyTorch / sklearn).

Replace the surrogate with the Chapter 2 GP before final thesis runs so that
the tube scheduling is driven by the *actual* identified uncertainty.
"""
import numpy as np


class GPUncertainty(object):
    def __init__(self, mode="regime", sigma2_base=1e-6, sigma2_dyn=5e-5,
                 pickle_path=""):
        self.mode = mode
        self.s2b = sigma2_base
        self.s2d = sigma2_dyn
        self.gp = None
        if mode == "trained":
            import pickle
            with open(pickle_path, "rb") as f:
                self.gp = pickle.load(f)   # [ADAPT to your GP object]

    def sigma_gp(self, x, u):
        """Return 12x12 posterior covariance at (x, u)."""
        if self.mode == "trained":
            return self._query_trained(x, u)
        s2 = self.s2b
        if self.mode == "regime":
            att = np.linalg.norm(x[3:5])          # |phi,theta|
            rate = np.linalg.norm(x[9:12])        # |p,q,r|
            s2 = self.s2b + self.s2d * (att / 0.3) ** 2 \
                             + self.s2d * (rate / 1.0) ** 2
        S = np.eye(12) * s2
        # weight attitude/rate channels more (they carry the dynamics)
        S[3:6, 3:6] *= 3.0
        S[9:12, 9:12] *= 3.0
        return S

    def _query_trained(self, x, u):
        """[ADAPT] Query your Chapter 2 GP. Expected: predictive covariance
        of the residual dynamics mapped into the 12-state space."""
        raise NotImplementedError(
            "Wire your trained GP here (see module docstring).")


def propagate_covariance(model, gp_unc, x0, u_hover, Np):
    """Sigma_{i+1} = A Sigma_i A^T + Sigma_GP,i   (thesis Eq. 3.42).

    Returns list [Sigma_0 .. Sigma_Np], Sigma_0 = small initial covariance.
    """
    Sigmas = [np.eye(12) * 1e-9]
    A, _ = model.jacobian_at(x0, u_hover)
    for i in range(Np):
        Sgp = gp_unc.sigma_gp(x0, u_hover)   # frozen-state surrogate
        Sigmas.append(A @ Sigmas[-1] @ A.T + Sgp)
    return Sigmas


def confidence_metric(Sigma):
    """gamma = sqrt(lambda_max(Sigma))   (thesis Eq. 3.44)."""
    ev = np.linalg.eigvalsh(Sigma)
    return float(np.sqrt(max(ev.max(), 0.0)))
