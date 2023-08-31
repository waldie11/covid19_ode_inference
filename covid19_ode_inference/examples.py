import sys

import matplotlib.pyplot as plt
import numpy as np

import pytensor.tensor as pt

import pymc as pm
import pymc.sampling.jax

import diffrax

import covid19_ode_inference as cov19_ode

from covid19_ode_inference.slow_modulation import get_cpkwargs

def model_cases_seropositivity_dead(
    N,
    cases_data,
    t_cases_data,
    seropos_data,
    t_seropos_data,
    dead_data,
    sim_model=False,
    fact_subs=4,
    cp_reporting_kwargs=dict(num_cps=3),
    cp_R_kwargs=dict(num_cps=8),
    cp_fatality_kwargs=dict(num_cps=1),
    truncater=pm.Bound, # pm.Truncated for toys
):
    end_sim = max(t_cases_data)

    t_solve_ODE = np.arange(-20, end_sim+fact_subs, fact_subs)

    coords = {
        **get_cpkwargs(cp_reporting_kwargs, t_cases_data, "reporting"),
        **get_cpkwargs(cp_R_kwargs, t_cases_data, "R"),
        **get_cpkwargs(cp_fatality_kwargs, t_cases_data, "fatality"),
        "t_solve_ODE": t_solve_ODE,
        "t_seropos_data": t_seropos_data,
        "t_cases_data": t_cases_data,
    }

    with pm.Model(coords=coords) as model:
        R0 = pm.LogNormal("R0", np.log(1), 1) if not sim_model else 1.2
        inv_gamma = (
            pm.Gamma("inv_gamma", alpha=40, beta=1 / 0.1)
            if not sim_model
            else pt.as_tensor_variable(4)
        )
        gamma = pm.Deterministic("gamma", 1 / inv_gamma)

        inv_epsilon = (
            pm.Gamma("inv_epsilon", alpha=2000, beta=1 / 0.1)
            if not sim_model
            else pt.as_tensor_variable(200)
        )
        epsilon = pm.Deterministic("epsilon", 1 / inv_epsilon)

        eta_base = pm.Normal("eta_base", 0, 1) if not sim_model else 0.3
        t_pos_rep, Delta_rhos_rep, transients_rep = cov19_ode.priors_for_cps(
            name="reporting",
            name_positions="t_pos_rep",
            name_magnitudes="Delta_rhos",
            name_durations="transients_rep",
            beta_magnitude=1,
            sigma_magnitude_fix=0.1 if sim_model else None,
        )

        eta_report = pt.sigmoid(
            eta_base
            + cov19_ode.slow_modulation.sigmoidal_changepoints(
                ts_out=t_cases_data,
                positions_cp=t_pos_rep,
                magnitudes_cp=Delta_rhos_rep,
                durations_cp=transients_rep,
            )
        )
        pm.Deterministic("eta_report", eta_report, dims=("t_cases_data",))

        # delta_base = pm.Normal("delta_base", 0, 1) if not sim_model else 0.3

        delta_base = pm.Cauchy("delta_base", -10, beta=1 ) if not sim_model else 0.025
        t_pos_fatality, Delta_rhos_fatality, transients_fatality = cov19_ode.priors_for_cps(
            name="fatality",
            name_positions="t_pos_fatality",
            name_magnitudes="Delta_rhos_fatality",
            name_durations="transients_fatality",
            beta_magnitude=1,
            sigma_magnitude_fix=0.1 if sim_model else None,
        )

        ifr_t = pt.sigmoid(
            delta_base
            + cov19_ode.slow_modulation.sigmoidal_changepoints_invt(
                ts_out=t_solve_ODE,
                positions_cp=t_pos_fatality,
                magnitudes_cp=Delta_rhos_fatality,
                durations_cp=transients_fatality,
            )
        )
        pm.Deterministic("ifr_t", ifr_t, dims=("t_solve_ODE",))

        t_pos_R, Delta_rhos_R, transients_R = cov19_ode.priors_for_cps(
            name="R",
            name_positions="t_pos_R",
            name_magnitudes="Delta_rhos_R",
            name_durations="transients_R",
            beta_magnitude=1,
            sigma_magnitude_fix=0.1 if sim_model else None,
        )

        reproduction_scale_t = pt.exp(
            cov19_ode.slow_modulation.sigmoidal_changepoints(
                ts_out=t_solve_ODE,
                positions_cp=t_pos_R,
                magnitudes_cp=Delta_rhos_R,
                durations_cp=transients_R,
            )
        )
        beta_t = R0 * gamma * reproduction_scale_t

        frac_S_0 = truncater("frac_S_0", dist=pm.Logistic.dist(100), lower=0, upper=100)/100 if not sim_model else 0.99
        frac_I_0 = truncater("frac_I_0", dist=pm.Logistic.dist(0), lower=0, upper=100)/100 if not sim_model else pt.clamp(100./N, min=0, max=1-frac_S_0)

        Naive_0 = frac_S_0 * N
        S_0 = 0.
        I_0 = frac_I_0 * N
        R_0 = pm.Deterministic("R_0", N - S_0 - I_0 - Naive_0)
        D_0 = truncater("D_0", dist=pm.Logistic.dist(0), lower=0) if not sim_model else 0.

        pm.Deterministic("beta_t", beta_t, dims=("t_solve_ODE",))

        def NSIRSD(t, y, args):
            Naive, S, I, R, D = y
            (β, δ), (γ, ε, Npop) = args
            # N = S + I + R
            dNaive = -β(t) * I * Naive / Npop
            dS = -β(t) * I * S / Npop + ε * R
            # dI = β(t) * I * S / N - γ * I - δ(t) * I
            dR = γ * I - ε * R
            dD = δ(t) * I
            dI = -dNaive -dS - dR - dD
            return dNaive, dS, dI, dR, dD

        integrator = cov19_ode.CompModelsIntegrator(
            ts_out=t_solve_ODE,
            t_0=min(t_solve_ODE),
            ts_solver=t_solve_ODE,
            ts_arg=t_solve_ODE,
            interp="cubic",
            solver=diffrax.Bosh3(),  # a 3rd order method
            adjoint=diffrax.RecursiveCheckpointAdjoint(checkpoints=len(t_solve_ODE)),
        )
        NSIRSD_integrator = integrator.get_Op(
            NSIRSD, return_shapes=[() for _ in range(5)], name="NSIRSD"
        )

        Naive, S, I, R, D = NSIRSD_integrator(
            y0=(Naive_0, S_0, I_0, R_0, D_0), arg_t=(beta_t, ifr_t), constant_args=(gamma, epsilon, N)
        )

        pm.Deterministic("Naive", S, dims=("t_solve_ODE",))
        pm.Deterministic("S", S, dims=("t_solve_ODE",))
        pm.Deterministic("I", I, dims=("t_solve_ODE",))
        pm.Deterministic("R", R, dims=("t_solve_ODE",))
        pm.Deterministic("D", D, dims=("t_solve_ODE",))

        new_positive = cov19_ode.interpolate(
            ts_in=t_solve_ODE,
            ts_out=t_cases_data,
            y=beta_t * (Naive+S) * I / N,
            ret_gradients=False,
            method="cubic",
        )

        pm.Deterministic("new_positive", new_positive)
        new_reported = new_positive * eta_report
        pm.Deterministic("new_reported", new_reported)

        error_rep = truncater("error_report", dist=pm.Logistic.dist(-1), lower=0)
        pm.Normal(
            "new_reported_data",
            new_reported,
            pt.sqrt(new_reported + 1e-5) * error_rep,
            observed=cases_data if not sim_model else None,
        )

        sero_at_data = cov19_ode.interpolate(
            ts_in=t_solve_ODE,
            ts_out=t_seropos_data,
            y=N-Naive,
            ret_gradients=False,
            method="cubic",
        )

        error_sero = pm.HalfNormal("error_sero", sigma=0.01)
        pm.Normal(
            "sero_data",
            sero_at_data / N,
            error_sero,
            observed=seropos_data if not sim_model else None,
        )

        sero_at_cases = cov19_ode.interpolate(
            ts_in=t_solve_ODE,
            ts_out=t_cases_data,
            y=N-Naive,
            ret_gradients=False,
            method="cubic",
        )

        pm.Deterministic("Sero_t", sero_at_cases / N)

        dead = cov19_ode.interpolate(
            ts_in=t_solve_ODE,
            ts_out=t_cases_data,
            y=D,
            ret_gradients=False,
            method="cubic",
        )

        error_dead = truncater("error_dead", dist=pm.Logistic.dist(-1), lower=0)
        pm.Normal(
            "dead_data",
            dead,
            pt.sqrt(dead + 1e-5) * error_dead,
            observed=dead_data if not sim_model else None,
        )

        pm.Deterministic("Dead_t", dead)
    return model
