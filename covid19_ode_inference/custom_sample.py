from pymc.sampling.mcmc import assign_step_methods, _check_start_shape
from typing import overload, Optional, Union, Sequence, Literal, Dict, Any
from pymc.initial_point import StartDict
from pymc.backends import TraceOrBackend, Model
from pymc.sampling.mcmc import MultiTrace, CompoundStep, PopulationArrayStepShared
from arviz import InferenceData
from numpy.random import RandomState
from pymc import modelcontext
from pymc.util import _get_seeds_per_chain
from pymc import _log
import pymc as pm
from pymc.sampling.mcmc import NUTS, init_nuts, init_traces, _print_step_hierarchy
import numpy as np
import time
import pickle
from pymc.sampling.mcmc import _sample_population, _sample_many, _mp_sample, _sample_return#, _sample_external_nuts

#numpyro deps
from pymc.sampling.jax import _update_numpyro_nuts_kwargs, get_default_varnames, _get_batched_jittered_initial_points, get_jaxified_graph, _postprocess_samples, get_jaxified_logp, _update_coords_and_dims, find_observations, find_constants, _sample_stats_to_xarray, make_attrs
import arviz as az
import warnings
from datetime import datetime
import sys
import jax
from functools import partial

def my_sample_numpyro_nuts(
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.8,
    random_seed: Optional[RandomState] = None,
    initvals: Optional[Union[StartDict, Sequence[Optional[StartDict]]]] = None,
    model: Optional[Model] = None,
    var_names: Optional[Sequence[str]] = None,
    progressbar: bool = True,
    keep_untransformed: bool = False,
    chain_method: str = "parallel",
    postprocessing_backend: Optional[str] = None,
    postprocessing_chunks: Optional[int] = None,
    idata_kwargs: Optional[Dict] = None,
    nuts_kwargs: Optional[Dict] = None,
) -> az.InferenceData:
    """
    Draw samples from the posterior using the NUTS method from the ``numpyro`` library.

    Parameters
    ----------
    draws : int, default 1000
        The number of samples to draw. The number of tuned samples are discarded by
        default.
    tune : int, default 1000
        Number of iterations to tune. Samplers adjust the step sizes, scalings or
        similar during tuning. Tuning samples will be drawn in addition to the number
        specified in the ``draws`` argument.
    chains : int, default 4
        The number of chains to sample.
    target_accept : float in [0, 1].
        The step size is tuned such that we approximate this acceptance rate. Higher
        values like 0.9 or 0.95 often work better for problematic posteriors.
    random_seed : int, RandomState or Generator, optional
        Random seed used by the sampling steps.
    initvals: StartDict or Sequence[Optional[StartDict]], optional
        Initial values for random variables provided as a dictionary (or sequence of
        dictionaries) mapping the random variable (by name or reference) to desired
        starting values.
    model : Model, optional
        Model to sample from. The model needs to have free random variables. When inside
        a ``with`` model context, it defaults to that model, otherwise the model must be
        passed explicitly.
    var_names : sequence of str, optional
        Names of variables for which to compute the posterior samples. Defaults to all
        variables in the posterior.
    progressbar : bool, default True
        Whether or not to display a progress bar in the command line. The bar shows the
        percentage of completion, the sampling speed in samples per second (SPS), and
        the estimated remaining time until completion ("expected time of arrival"; ETA).
    keep_untransformed : bool, default False
        Include untransformed variables in the posterior samples. Defaults to False.
    chain_method : str, default "parallel"
        Specify how samples should be drawn. The choices include "sequential",
        "parallel", and "vectorized".
    postprocessing_backend : Optional[str]
        Specify how postprocessing should be computed. gpu or cpu
    postprocessing_chunks: Optional[int], default None
        Specify the number of chunks the postprocessing should be computed in. More
        chunks reduces memory usage at the cost of losing some vectorization, None
        uses jax.vmap
    idata_kwargs : dict, optional
        Keyword arguments for :func:`arviz.from_dict`. It also accepts a boolean as
        value for the ``log_likelihood`` key to indicate that the pointwise log
        likelihood should not be included in the returned object. Values for
        ``observed_data``, ``constant_data``, ``coords``, and ``dims`` are inferred from
        the ``model`` argument if not provided in ``idata_kwargs``. If ``coords`` and
        ``dims`` are provided, they are used to update the inferred dictionaries.
    nuts_kwargs: dict, optional
        Keyword arguments for :func:`numpyro.infer.NUTS`.

    Returns
    -------
    InferenceData
        ArviZ ``InferenceData`` object that contains the posterior samples, together
        with their respective sample stats and pointwise log likeihood values (unless
        skipped with ``idata_kwargs``).
    """

    import numpyro

    from numpyro.infer import MCMC, NUTS

    model = modelcontext(model)

    if var_names is None:
        var_names = model.unobserved_value_vars

    vars_to_sample = list(get_default_varnames(var_names, include_transformed=keep_untransformed))

    coords = {
        cname: np.array(cvals) if isinstance(cvals, tuple) else cvals
        for cname, cvals in model.coords.items()
        if cvals is not None
    }

    dims = {
        var_name: [dim for dim in dims if dim is not None]
        for var_name, dims in model.named_vars_to_dims.items()
    }

    (random_seed,) = _get_seeds_per_chain(random_seed, 1)

    tic1 = datetime.now()
    print("Compiling...", file=sys.stdout)

    init_params = _get_batched_jittered_initial_points(
        model=model,
        chains=chains,
        initvals=initvals,
        random_seed=random_seed,
    )

    logp_fn = get_jaxified_logp(model, negative_logp=False)

    nuts_kwargs = _update_numpyro_nuts_kwargs(nuts_kwargs)
    nuts_kernel = NUTS(
        potential_fn=logp_fn,
        target_accept_prob=target_accept,
        **nuts_kwargs,
    )

    pmap_numpyro = MCMC(
        nuts_kernel,
        num_warmup=tune,
        num_samples=draws,
        num_chains=chains,
        postprocess_fn=None,
        chain_method=chain_method,
        progress_bar=progressbar,
        jit_model_args=True,
    )
    # mcmc = MCMC(NUTS(model), num_warmup=100, num_samples=100)
    # mcmc.run(random.PRNGKey(0))
    # first_100_samples = mcmc.get_samples()
    # mcmc.post_warmup_state = mcmc.last_state
    # mcmc.run(mcmc.post_warmup_state.rng_key)  # or mcmc.run(random.PRNGKey(1))
    # second_100_samples = mcmc.get_samples()

    tic2 = datetime.now()
    print("Compilation time = ", tic2 - tic1, file=sys.stdout)

    print("Sampling...", file=sys.stdout)

    map_seed = jax.random.PRNGKey(random_seed)
    if chains > 1:
        map_seed = jax.random.split(map_seed, chains)

    pmap_numpyro.warmup(
        map_seed,
        init_params=init_params,
        collect_warmup=True,
        extra_fields=(
            "num_steps",
            "potential_energy",
            "energy",
            "adapt_state.step_size",
            "accept_prob",
            "diverging",
        ),
    )

    raw_mcmc_warmup_samples = pmap_numpyro.get_samples(group_by_chain=True)

    warmup_sample_stats = _sample_stats_to_xarray(pmap_numpyro)

    pmap_numpyro.post_warmup_state = pmap_numpyro.last_state
    pmap_numpyro.run(
        pmap_numpyro.post_warmup_state.rng_key,
        extra_fields=(
            "num_steps",
            "potential_energy",
            "energy",
            "adapt_state.step_size",
            "accept_prob",
            "diverging",
        ),
    )

    raw_mcmc_samples = pmap_numpyro.get_samples(group_by_chain=True)

    tic3 = datetime.now()
    print("Sampling time = ", tic3 - tic2, file=sys.stdout)

    print("Transforming variables...", file=sys.stdout)
    jax_fn = get_jaxified_graph(inputs=model.value_vars, outputs=vars_to_sample)
    result_warmup = _postprocess_samples(
        jax_fn, raw_mcmc_warmup_samples, postprocessing_backend, num_chunks=postprocessing_chunks
    )
    result = _postprocess_samples(
        jax_fn, raw_mcmc_samples, postprocessing_backend, num_chunks=postprocessing_chunks
    )
    mcmc_samples = {v.name: r for v, r in zip(vars_to_sample, result)}

    tic4 = datetime.now()
    print("Transformation time = ", tic4 - tic3, file=sys.stdout)

    if idata_kwargs is None:
        idata_kwargs = {}
    else:
        idata_kwargs = idata_kwargs.copy()

    if idata_kwargs.pop("log_likelihood", False):
        tic5 = datetime.now()
        print("Computing Log Likelihood...", file=sys.stdout)
        log_likelihood = _get_log_likelihood(
            model,
            raw_mcmc_samples,
            backend=postprocessing_backend,
            num_chunks=postprocessing_chunks,
        )
        tic6 = datetime.now()
        print("Log Likelihood time = ", tic6 - tic5, file=sys.stdout)
    else:
        log_likelihood = None

    attrs = {
        "sampling_time": (tic3 - tic2).total_seconds(),
    }

    posterior = mcmc_samples
    # Update 'coords' and 'dims' extracted from the model with user 'idata_kwargs'
    # and drop keys 'coords' and 'dims' from 'idata_kwargs' if present.
    _update_coords_and_dims(coords=coords, dims=dims, idata_kwargs=idata_kwargs)
    # Use 'partial' to set default arguments before passing 'idata_kwargs'
    to_trace = partial(
        az.from_dict,
        log_likelihood=log_likelihood,
        observed_data=find_observations(model),
        constant_data=find_constants(model),
        sample_stats=_sample_stats_to_xarray(pmap_numpyro),
        coords=coords,
        dims=dims,
        attrs=make_attrs(attrs, library=numpyro),
    )
    print(idata_kwargs)
    az_trace = to_trace(posterior=posterior, warmup_sample_stats=warmup_sample_stats, **idata_kwargs)
    return az_trace

def _my_sample_external_nuts(
    sampler: str,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    random_seed: Union[RandomState, None],
    initvals: Union[StartDict, Sequence[Optional[StartDict]], None],
    model: Model,
    progressbar: bool,
    idata_kwargs: Optional[Dict],
    nuts_sampler_kwargs: Optional[Dict],
    **kwargs,
):
    warnings.warn("Use of external NUTS sampler is still experimental", UserWarning)

    if nuts_sampler_kwargs is None:
        nuts_sampler_kwargs = {}

    if sampler == "nutpie":
        try:
            import nutpie
        except ImportError as err:
            raise ImportError(
                "nutpie not found. Install it with conda install -c conda-forge nutpie"
            ) from err

        if initvals is not None:
            warnings.warn(
                "`initvals` are currently not passed to nutpie sampler. "
                "Use `init_mean` kwarg following nutpie specification instead.",
                UserWarning,
            )

        if idata_kwargs is not None:
            warnings.warn(
                "`idata_kwargs` are currently ignored by the nutpie sampler",
                UserWarning,
            )

        compiled_model = nutpie.compile_pymc_model(model)
        idata = nutpie.sample(
            compiled_model,
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            seed=_get_seeds_per_chain(random_seed, 1)[0],
            progress_bar=progressbar,
            **nuts_sampler_kwargs,
        )
        return idata

    elif sampler == "numpyro":
        import pymc.sampling.jax as pymc_jax

        idata = pymc_jax.sample_numpyro_nuts(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            initvals=initvals,
            model=model,
            progressbar=progressbar,
            idata_kwargs=idata_kwargs,
            **nuts_sampler_kwargs,
        )
        return idata

    elif sampler == "mynumpyro":
        import pymc.sampling.jax as pymc_jax

        idata = my_sample_numpyro_nuts(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            initvals=initvals,
            model=model,
            progressbar=progressbar,
            idata_kwargs=idata_kwargs,
            **nuts_sampler_kwargs,
        )
        return idata

    elif sampler == "blackjax":
        import pymc.sampling.jax as pymc_jax

        idata = pymc_jax.sample_blackjax_nuts(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            initvals=initvals,
            model=model,
            idata_kwargs=idata_kwargs,
            **nuts_sampler_kwargs,
        )
        return idata

    else:
        raise ValueError(
            f"Sampler {sampler} not found. Choose one of ['nutpie', 'numpyro', 'blackjax', 'pymc']."
        )

@overload
def my_sample(
    draws: int = 1000,
    *,
    tune: int = 1000,
    chains: Optional[int] = None,
    cores: Optional[int] = None,
    random_seed: RandomState = None,
    progressbar: bool = True,
    step=None,
    nuts_sampler: str = "pymc",
    initvals: Optional[Union[StartDict, Sequence[Optional[StartDict]]]] = None,
    init: str = "auto",
    jitter_max_retries: int = 10,
    n_init: int = 200_000,
    trace: Optional[TraceOrBackend] = None,
    discard_tuned_samples: bool = True,
    compute_convergence_checks: bool = True,
    keep_warning_stat: bool = False,
    return_inferencedata: Literal[True] = True,
    idata_kwargs: Optional[Dict[str, Any]] = None,
    nuts_sampler_kwargs: Optional[Dict[str, Any]] = None,
    callback=None,
    mp_ctx=None,
    **kwargs,
) -> InferenceData:
    ...


@overload
def my_sample(
    draws: int = 1000,
    *,
    tune: int = 1000,
    chains: Optional[int] = None,
    cores: Optional[int] = None,
    random_seed: RandomState = None,
    progressbar: bool = True,
    step=None,
    nuts_sampler: str = "pymc",
    initvals: Optional[Union[StartDict, Sequence[Optional[StartDict]]]] = None,
    init: str = "auto",
    jitter_max_retries: int = 10,
    n_init: int = 200_000,
    trace: Optional[TraceOrBackend] = None,
    discard_tuned_samples: bool = True,
    compute_convergence_checks: bool = True,
    keep_warning_stat: bool = False,
    return_inferencedata: Literal[False],
    idata_kwargs: Optional[Dict[str, Any]] = None,
    nuts_sampler_kwargs: Optional[Dict[str, Any]] = None,
    callback=None,
    mp_ctx=None,
    model: Optional[Model] = None,
    **kwargs,
) -> MultiTrace:
    ...


# [docs]
def my_sample(
    draws: int = 1000,
    *,
    tune: int = 1000,
    chains: Optional[int] = None,
    cores: Optional[int] = None,
    random_seed: RandomState = None,
    progressbar: bool = True,
    step=None,
    nuts_sampler: str = "pymc",
    initvals: Optional[Union[StartDict, Sequence[Optional[StartDict]]]] = None,
    init: str = "auto",
    jitter_max_retries: int = 10,
    n_init: int = 200_000,
    trace: Optional[TraceOrBackend] = None,
    discard_tuned_samples: bool = True,
    compute_convergence_checks: bool = True,
    keep_warning_stat: bool = False,
    return_inferencedata: bool = True,
    idata_kwargs: Optional[Dict[str, Any]] = None,
    nuts_sampler_kwargs: Optional[Dict[str, Any]] = None,
    callback=None,
    mp_ctx=None,
    model: Optional[Model] = None,
    **kwargs,
) -> Union[InferenceData, MultiTrace]:
    r"""Draw samples from the posterior using the given step methods.

    Multiple step methods are supported via compound step methods.

    Parameters
    ----------
    draws : int
        The number of samples to draw. Defaults to 1000. The number of tuned samples are discarded
        by default. See ``discard_tuned_samples``.
    tune : int
        Number of iterations to tune, defaults to 1000. Samplers adjust the step sizes, scalings or
        similar during tuning. Tuning samples will be drawn in addition to the number specified in
        the ``draws`` argument, and will be discarded unless ``discard_tuned_samples`` is set to
        False.
    chains : int
        The number of chains to sample. Running independent chains is important for some
        convergence statistics and can also reveal multiple modes in the posterior. If ``None``,
        then set to either ``cores`` or 2, whichever is larger.
    cores : int
        The number of chains to run in parallel. If ``None``, set to the number of CPUs in the
        system, but at most 4.
    random_seed : int, array-like of int, RandomState or Generator, optional
        Random seed(s) used by the sampling steps. If a list, tuple or array of ints
        is passed, each entry will be used to seed each chain. A ValueError will be
        raised if the length does not match the number of chains.
    progressbar : bool, optional default=True
        Whether or not to display a progress bar in the command line. The bar shows the percentage
        of completion, the sampling speed in samples per second (SPS), and the estimated remaining
        time until completion ("expected time of arrival"; ETA).
        Only applicable to the pymc nuts sampler.
    step : function or iterable of functions
        A step function or collection of functions. If there are variables without step methods,
        step methods for those variables will be assigned automatically. By default the NUTS step
        method will be used, if appropriate to the model.
    nuts_sampler : str
        Which NUTS implementation to run. One of ["pymc", "nutpie", "blackjax", "numpyro"].
        This requires the chosen sampler to be installed.
        All samplers, except "pymc", require the full model to be continuous.
    initvals : optional, dict, array of dict
        Dict or list of dicts with initial value strategies to use instead of the defaults from
        `Model.initial_values`. The keys should be names of transformed random variables.
        Initialization methods for NUTS (see ``init`` keyword) can overwrite the default.
    init : str
        Initialization method to use for auto-assigned NUTS samplers. See `pm.init_nuts` for a list
        of all options. This argument is ignored when manually passing the NUTS step method.
        Only applicable to the pymc nuts sampler.
    jitter_max_retries : int
        Maximum number of repeated attempts (per chain) at creating an initial matrix with uniform
        jitter that yields a finite probability. This applies to ``jitter+adapt_diag`` and
        ``jitter+adapt_full`` init methods.
    n_init : int
        Number of iterations of initializer. Only works for 'ADVI' init methods.
    trace : backend, optional
        A backend instance or None.
        If None, the NDArray backend is used.
    discard_tuned_samples : bool
        Whether to discard posterior samples of the tune interval.
    compute_convergence_checks : bool, default=True
        Whether to compute sampler statistics like Gelman-Rubin and ``effective_n``.
    keep_warning_stat : bool
        If ``True`` the "warning" stat emitted by, for example, HMC samplers will be kept
        in the returned ``idata.sample_stat`` group.
        This leads to the ``idata`` not supporting ``.to_netcdf()`` or ``.to_zarr()`` and
        should only be set to ``True`` if you intend to use the "warning" objects right away.
        Defaults to ``False`` such that ``pm.drop_warning_stat`` is applied automatically,
        making the ``InferenceData`` compatible with saving.
    return_inferencedata : bool
        Whether to return the trace as an :class:`arviz:arviz.InferenceData` (True) object or a
        `MultiTrace` (False). Defaults to `True`.
    idata_kwargs : dict, optional
        Keyword arguments for :func:`pymc.to_inference_data`
    nuts_sampler_kwargs : dict, optional
        Keyword arguments for the sampling library that implements nuts.
        Only used when an external sampler is specified via the `nuts_sampler` kwarg.
    callback : function, default=None
        A function which gets called for every sample from the trace of a chain. The function is
        called with the trace and the current draw and will contain all samples for a single trace.
        the ``draw.chain`` argument can be used to determine which of the active chains the sample
        is drawn from.
        Sampling can be interrupted by throwing a ``KeyboardInterrupt`` in the callback.
    mp_ctx : multiprocessing.context.BaseContent
        A multiprocessing context for parallel sampling.
        See multiprocessing documentation for details.
    model : Model (optional if in ``with`` context)
        Model to sample from. The model needs to have free random variables.

    Returns
    -------
    trace : pymc.backends.base.MultiTrace or arviz.InferenceData
        A ``MultiTrace`` or ArviZ ``InferenceData`` object that contains the samples.

    Notes
    -----
    Optional keyword arguments can be passed to ``sample`` to be delivered to the
    ``step_method``\ s used during sampling.

    For example:

       1. ``target_accept`` to NUTS: nuts={'target_accept':0.9}
       2. ``transit_p`` to BinaryGibbsMetropolis: binary_gibbs_metropolis={'transit_p':.7}

    Note that available step names are:

    ``nuts``, ``hmc``, ``metropolis``, ``binary_metropolis``,
    ``binary_gibbs_metropolis``, ``categorical_gibbs_metropolis``,
    ``DEMetropolis``, ``DEMetropolisZ``, ``slice``

    The NUTS step method has several options including:

        * target_accept : float in [0, 1]. The step size is tuned such that we
          approximate this acceptance rate. Higher values like 0.9 or 0.95 often
          work better for problematic posteriors. This argument can be passed directly to sample.
        * max_treedepth : The maximum depth of the trajectory tree
        * step_scale : float, default 0.25
          The initial guess for the step size scaled down by :math:`1/n**(1/4)`,
          where n is the dimensionality of the parameter space

    Alternatively, if you manually declare the ``step_method``\ s, within the ``step``
       kwarg, then you can address the ``step_method`` kwargs directly.
       e.g. for a CompoundStep comprising NUTS and BinaryGibbsMetropolis,
       you could send ::

        step=[pm.NUTS([freeRV1, freeRV2], target_accept=0.9),
              pm.BinaryGibbsMetropolis([freeRV3], transit_p=.7)]

    You can find a full list of arguments in the docstring of the step methods.

    Examples
    --------
    .. code:: ipython

        In [1]: import pymc as pm
           ...: n = 100
           ...: h = 61
           ...: alpha = 2
           ...: beta = 2

        In [2]: with pm.Model() as model: # context management
           ...:     p = pm.Beta("p", alpha=alpha, beta=beta)
           ...:     y = pm.Binomial("y", n=n, p=p, observed=h)
           ...:     idata = pm.sample()

        In [3]: az.summary(idata, kind="stats")

        Out[3]:
            mean     sd  hdi_3%  hdi_97%
        p  0.609  0.047   0.528    0.699
    """
    if "start" in kwargs:
        if initvals is not None:
            raise ValueError("Passing both `start` and `initvals` is not supported.")
        warnings.warn(
            "The `start` kwarg was renamed to `initvals` and can now do more. Please check the docstring.",
            FutureWarning,
            stacklevel=2,
        )
        initvals = kwargs.pop("start")
    if nuts_sampler_kwargs is None:
        nuts_sampler_kwargs = {}
    if "target_accept" in kwargs:
        if "nuts" in kwargs and "target_accept" in kwargs["nuts"]:
            raise ValueError(
                "`target_accept` was defined twice. Please specify it either as a direct keyword argument or in the `nuts` kwarg."
            )
        if "nuts" in kwargs:
            kwargs["nuts"]["target_accept"] = kwargs.pop("target_accept")
        else:
            kwargs["nuts"] = {"target_accept": kwargs.pop("target_accept")}
    if isinstance(trace, list):
        raise DeprecationWarning(
            "We have removed support for partial traces because it simplified things."
            " Please open an issue if & why this is a problem for you."
        )

    model = modelcontext(model)
    if not model.free_RVs:
        raise SamplingError(
            "Cannot sample from the model, since the model does not contain any free variables."
        )

    if cores is None:
        cores = min(4, _cpu_count())

    if chains is None:
        chains = max(2, cores)

    if random_seed == -1:
        random_seed = None
    random_seed_list = _get_seeds_per_chain(random_seed, chains)

    if not discard_tuned_samples and not return_inferencedata:
        warnings.warn(
            "Tuning samples will be included in the returned `MultiTrace` object, which can lead to"
            " complications in your downstream analysis. Please consider to switch to `InferenceData`:\n"
            "`pm.sample(..., return_inferencedata=True)`",
            UserWarning,
            stacklevel=2,
        )

    # small trace warning
    if draws == 0:
        msg = "Tuning was enabled throughout the whole trace."
        _log.warning(msg)
    elif draws < 500:
        msg = "Only %s samples in chain." % draws
        _log.warning(msg)

    auto_nuts_init = True
    if step is not None:
        if isinstance(step, CompoundStep):
            for method in step.methods:
                if isinstance(method, NUTS):
                    auto_nuts_init = False
        elif isinstance(step, NUTS):
            auto_nuts_init = False

    initial_points = None
    step = assign_step_methods(model, step, methods=pm.STEP_METHODS, step_kwargs=kwargs)

    if nuts_sampler != "pymc":
        if not isinstance(step, NUTS):
            raise ValueError(
                "Model can not be sampled with NUTS alone. Your model is probably not continuous."
            )
        return _my_sample_external_nuts(
            sampler=nuts_sampler,
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=kwargs.pop("nuts", {}).get("target_accept", 0.8),
            random_seed=random_seed,
            initvals=initvals,
            model=model,
            progressbar=progressbar,
            idata_kwargs=idata_kwargs,
            nuts_sampler_kwargs=nuts_sampler_kwargs,
            **kwargs,
        )

    if isinstance(step, list):
        step = CompoundStep(step)
    elif isinstance(step, NUTS) and auto_nuts_init:
        if "nuts" in kwargs:
            nuts_kwargs = kwargs.pop("nuts")
            [kwargs.setdefault(k, v) for k, v in nuts_kwargs.items()]
        _log.info("Auto-assigning NUTS sampler...")
        initial_points, step = init_nuts(
            init=init,
            chains=chains,
            n_init=n_init,
            model=model,
            random_seed=random_seed_list,
            progressbar=progressbar,
            jitter_max_retries=jitter_max_retries,
            tune=tune,
            initvals=initvals,
            **kwargs,
        )

    if initial_points is None:
        # Time to draw/evaluate numeric start points for each chain.
        ipfns = make_initial_point_fns_per_chain(
            model=model,
            overrides=initvals,
            jitter_rvs=set(),
            chains=chains,
        )
        initial_points = [ipfn(seed) for ipfn, seed in zip(ipfns, random_seed_list)]

    # One final check that shapes and logps at the starting points are okay.
    ip: Dict[str, np.ndarray]
    for ip in initial_points:
        model.check_start_vals(ip)
        _check_start_shape(model, ip)

    # Create trace backends for each chain
    run, traces = init_traces(
        backend=trace,
        chains=chains,
        expected_length=draws + tune,
        step=step,
        initial_point=ip,
        model=model,
    )

    sample_args = {
        "draws": draws + tune,  # FIXME: Why is tune added to draws?
        "step": step,
        "start": initial_points,
        "traces": traces,
        "chains": chains,
        "tune": tune,
        "progressbar": progressbar,
        "model": model,
        "cores": cores,
        "callback": callback,
        "discard_tuned_samples": discard_tuned_samples,
    }
    parallel_args = {
        "mp_ctx": mp_ctx,
    }

    sample_args.update(kwargs)

    has_population_samplers = np.any(
        [
            isinstance(m, PopulationArrayStepShared)
            for m in (step.methods if isinstance(step, CompoundStep) else [step])
        ]
    )

    parallel = cores > 1 and chains > 1 and not has_population_samplers
    # At some point it was decided that PyMC should not set a global seed by default,
    # unless the user specified a seed. This is a symptom of the fact that PyMC samplers
    # are built around global seeding. This branch makes sure we maintain this unspoken
    # rule. See https://github.com/pymc-devs/pymc/pull/1395.
    if parallel:
        # For parallel sampling we can pass the list of random seeds directly, as
        # global seeding will only be called inside each process
        sample_args["random_seed"] = random_seed_list
    else:
        # We pass None if the original random seed was None. The single core sampler
        # methods will only set a global seed when it is not None.
        sample_args["random_seed"] = random_seed if random_seed is None else random_seed_list

    t_start = time.time()
    if parallel:
        _log.info(f"Multiprocess sampling ({chains} chains in {cores} jobs)")
        _print_step_hierarchy(step)
        try:
            _mp_sample(**sample_args, **parallel_args)
        except pickle.PickleError:
            _log.warning("Could not pickle model, sampling singlethreaded.")
            _log.debug("Pickling error:", exc_info=True)
            parallel = False
        except AttributeError as e:
            if not str(e).startswith("AttributeError: Can't pickle"):
                raise
            _log.warning("Could not pickle model, sampling singlethreaded.")
            _log.debug("Pickling error:", exc_info=True)
            parallel = False
    if not parallel:
        if has_population_samplers:
            _log.info(f"Population sampling ({chains} chains)")
            _print_step_hierarchy(step)
            _sample_population(initial_points=initial_points, parallelize=cores > 1, **sample_args)
        else:
            _log.info(f"Sequential sampling ({chains} chains in 1 job)")
            _print_step_hierarchy(step)
            _sample_many(**sample_args)

    t_sampling = time.time() - t_start

    # Packaging, validating and returning the result was extracted
    # into a function to make it easier to test and refactor.
    return _sample_return(
        run=run,
        traces=traces,
        tune=tune,
        t_sampling=t_sampling,
        discard_tuned_samples=discard_tuned_samples,
        compute_convergence_checks=compute_convergence_checks,
        return_inferencedata=return_inferencedata,
        keep_warning_stat=keep_warning_stat,
        idata_kwargs=idata_kwargs or {},
        model=model,
    )
