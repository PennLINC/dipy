"""Analytic DTI/DKI and MAP-MRI/SHORE scalars for FORCE library entries.

Every FORCE library entry is an *exact, known* mixture of Gaussian
compartments::

    S(b, g) / S0 = sum_c w_c exp(-b g^T D_c g),   sum_c w_c = 1

so its diffusion tensor, diffusional-kurtosis tensor and q-space indices all
have closed forms in the compartment tensors ``{w_c, D_c}`` -- they do **not**
require fitting the (arbitrarily sampled) signal.  This makes every stored
scalar independent of the acquisition scheme (single-shell, cartesian, random
all give identical results) and removes the shell-finding that broke
:func:`dipy.sims.force.generate_force_simulations` on non-shelled data.

The FORCE Cython generator emits, per voxel, two orientation-weight vectors
``intra_odf`` and ``extra_odf`` (defined on ``sphere`` vertices) plus the raw
compartment diffusivities.  From these this module reconstructs everything:

* the mean diffusion tensor ``D_app`` and the 4th-order diffusion covariance
  ``C`` (-> cumulant DTI/DKI, closed form),
* the noise-free signal on any gradient table (-> a scheme-independent
  "canonical" DTI/DKI/MAP-MRI fit, and NG/PA which need basis coefficients),
* the closed-form MAP-MRI indices RTOP/RTAP/RTPP/MSD/QIV.

All heavy numerics live here in pure NumPy; the public analytic scalar
functions of :mod:`dipy.reconst.dti`, :mod:`dipy.reconst.dki` and
:mod:`dipy.reconst.mapmri` are *imported and reused* -- those modules are not
modified.

References
----------
Multi-Gaussian cumulant DKI: Jensen & Helpern 2010; the diffusion covariance
tensor / QTI: Westin 2016.  MAP-MRI indices: Ozarslan 2013; Fick 2016.
Propagator anisotropy: TORTOISE v4 (Avram/Ozarslan).
"""

import numpy as np

# ---------------------------------------------------------------------------
# kurtosis-tensor element ordering used by dipy.reconst.dki (see Wrotate)
# ---------------------------------------------------------------------------
_KT_INDS = (
    (0, 0, 0, 0), (1, 1, 1, 1), (2, 2, 2, 2),
    (0, 0, 0, 1), (0, 0, 0, 2), (0, 1, 1, 1),
    (1, 1, 1, 2), (0, 2, 2, 2), (1, 2, 2, 2),
    (0, 0, 1, 1), (0, 0, 2, 2), (1, 1, 2, 2),
    (0, 0, 1, 2), (0, 1, 1, 2), (0, 1, 2, 2),
)


def _check_mixture_weights(total, *, atol=1e-2):
    """Validate that a mixture's total compartment weight is ~1 per voxel.

    ``C = <D(x)D> - <D>(x)<D>`` is only a valid covariance (and RTOP/RTAP/... a
    valid EAP) when the compartment weights sum to 1.  Unlike
    :func:`dki_params_from_tensor_distribution` (which renormalizes), the
    from-odf helpers assume a *complete, normalized* mixture, so validate it.
    """
    total = np.asarray(total, np.float64)
    if not np.all(np.abs(total - 1.0) <= atol):
        worst = float(np.max(np.abs(total - 1.0)))
        raise ValueError(
            "FORCE mixture weights must sum to 1 per voxel "
            f"(intra + extra + gm_frac + csf_frac); worst deviation {worst:.3g}. "
            "Pass a complete, normalized mixture."
        )


# ---------------------------------------------------------------------------
# cumulant DTI/DKI from the moment tensors
# ---------------------------------------------------------------------------
def _full_symmetrize(C):
    """Full index symmetrization of an elasticity-symmetric rank-4 tensor.

    ``C`` has minor+major symmetries; the diffusional kurtosis tensor is fully
    symmetric, so ``W ~ (C_ijkl + C_ikjl + C_iljk) / 3``.  Works on a leading
    batch axis: ``C`` has shape ``(..., 3, 3, 3, 3)``.
    """
    a = C.ndim - 4  # first tensor axis
    t2 = np.moveaxis(C, a + 2, a + 1)             # C_ikjl
    t3 = np.moveaxis(np.moveaxis(C, a + 2, a + 1), a + 3, a + 2)  # C_iljk
    return (C + t2 + t3) / 3.0


def dki_scalars_from_params(dki_params):
    """Return a dict of DTI/DKI scalars from ``dki_params`` (..., 27).

    Reuses the public analytic scalar functions of :mod:`dipy.reconst.dti` and
    :mod:`dipy.reconst.dki` (those modules are not modified).
    """
    from dipy.reconst import dki, dti

    ev = dki_params[..., :3]
    return {
        "fa": dti.fractional_anisotropy(ev),
        "md": dti.mean_diffusivity(ev),
        "rd": dti.radial_diffusivity(ev),
        "ad": dti.axial_diffusivity(ev),
        "ak": dki.axial_kurtosis(dki_params),
        "rk": dki.radial_kurtosis(dki_params),
        "mk": dki.mean_kurtosis(dki_params),
        "mkt": dki.mean_kurtosis_tensor(dki_params),
        "kfa": dki.kurtosis_fractional_anisotropy(dki_params),
    }


def dki_params_from_tensor_distribution(weights, tensors):
    """dipy ``dki_params`` (27,) for a weighted set of Gaussian compartments.

    Parameters
    ----------
    weights : ndarray (n_comp,)
        Compartment volume fractions (need not be normalized).
    tensors : ndarray (n_comp, 3, 3)
        Compartment diffusion tensors.

    Returns
    -------
    dki_params : ndarray (27,)
    """
    w = np.asarray(weights, np.float64)
    w = w / w.sum()
    T = np.asarray(tensors, np.float64)
    D_app = np.einsum("c,cij->ij", w, T)
    DD = np.einsum("c,cij,ckl->ijkl", w, T, T)
    C = DD - np.einsum("ij,kl->ijkl", D_app, D_app)
    return dki_params_from_moments(D_app, C)


def dki_params_from_moments(D_app, C):
    """Assemble dipy ``dki_params`` (..., 27) from the mean tensor and
    covariance of a Gaussian-compartment mixture.

    Parameters
    ----------
    D_app : ndarray (..., 3, 3)
        Mean diffusion tensor ``sum_c w_c D_c``.
    C : ndarray (..., 3, 3, 3, 3)
        Diffusion covariance ``sum_c w_c D_c(x)D_c - D_app(x)D_app``.

    Returns
    -------
    dki_params : ndarray (..., 27)
        ``[evals(3), evecs(9, row-major), kt(15)]`` in the standard Cartesian
        frame, directly consumable by :mod:`dipy.reconst.dki` scalar functions.
    """
    D_app = np.asarray(D_app, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    batch = D_app.shape[:-2]
    Dm = D_app.reshape(-1, 3, 3)
    Cm = C.reshape(-1, 3, 3, 3, 3)
    n = Dm.shape[0]

    MD = np.trace(Dm, axis1=1, axis2=2) / 3.0            # (n,)
    symC = _full_symmetrize(Cm)                          # (n,3,3,3,3)
    # guard against a degenerate (zero-trace) mean tensor; unreachable for
    # physical diffusivities but avoids inf/nan in the W scaling.
    md2 = np.maximum(MD, 1e-10) ** 2
    W = 3.0 * symC / md2[:, None, None, None, None]
    kt = np.stack([W[:, i, j, k, l] for (i, j, k, l) in _KT_INDS], axis=1)  # (n,15)

    evals, evecs = np.linalg.eigh(Dm)                    # ascending
    order = np.argsort(evals, axis=1)[:, ::-1]
    evals = np.take_along_axis(evals, order, axis=1)
    evecs = np.take_along_axis(evecs, order[:, None, :], axis=2)  # cols=eigvecs

    params = np.concatenate(
        [evals, evecs.reshape(n, 9), kt], axis=1
    )
    return params.reshape(batch + (27,))


# ---------------------------------------------------------------------------
# reconstruct the mixture from the emitted intra/extra orientation weights
# ---------------------------------------------------------------------------
def orientation_moment_tensors(verts):
    """Return per-vertex ``vv^T`` (M,3,3) and ``(vv^T)(x)(vv^T)`` (M,3,3,3,3)."""
    verts = np.ascontiguousarray(verts, dtype=np.float64)
    VV = np.einsum("vi,vj->vij", verts, verts)
    VV4 = np.einsum("vij,vkl->vijkl", VV, VV)
    return VV, VV4


def moments_from_odfs(
    intra_odf, extra_odf, verts, d_par, d_perp,
    gm_frac, gm_d, csf_frac, csf_d, *, soma_frac=0.0, soma_d=1.0e-3,
    intra_dperp=0.0, VV=None, VV4=None,
):
    """Mean tensor ``D_app`` and covariance ``C`` of the mixture.

    ``intra_odf`` / ``extra_odf`` are the (fibre-fraction weighted) orientation
    weights of the intra-axonal (stick) and extra-axonal (zeppelin)
    compartments on ``verts``; each row sums to ``wm_fraction * <compartment
    fraction>``.  Scalars may be arrays (one per voxel).  Set ``intra_dperp`` to
    a small floor to regularize otherwise-degenerate sticks (only for q-space
    indices; leave 0 for the true cumulant DKI).

    ``soma_frac`` / ``soma_d`` add an optional isotropic *soma/dot* compartment
    (a low-diffusivity isotropic Gaussian; ``soma_d -> 0`` is the dot limit).
    With ``soma_frac=0`` (the default) it contributes nothing.
    """
    intra_odf = np.atleast_2d(np.asarray(intra_odf, np.float64))
    extra_odf = np.atleast_2d(np.asarray(extra_odf, np.float64))
    n = intra_odf.shape[0]
    if VV is None or VV4 is None:
        VV, VV4 = orientation_moment_tensors(verts)
    d_par = np.broadcast_to(np.asarray(d_par, np.float64), (n,))
    d_perp = np.broadcast_to(np.asarray(d_perp, np.float64), (n,))
    gm_frac = np.broadcast_to(np.asarray(gm_frac, np.float64), (n,))
    gm_d = np.broadcast_to(np.asarray(gm_d, np.float64), (n,))
    csf_frac = np.broadcast_to(np.asarray(csf_frac, np.float64), (n,))
    csf_d = np.broadcast_to(np.asarray(csf_d, np.float64), (n,))
    soma_frac = np.broadcast_to(np.asarray(soma_frac, np.float64), (n,))
    soma_d = np.broadcast_to(np.asarray(soma_d, np.float64), (n,))
    ip = np.broadcast_to(np.asarray(intra_dperp, np.float64), (n,))

    I3 = np.eye(3)
    IxI = np.einsum("ij,kl->ijkl", I3, I3)

    # orientation moments weighted by each compartment's odf
    Om_in = np.einsum("nv,vij->nij", intra_odf, VV)      # sum p_v vv^T
    Om_ex = np.einsum("nv,vij->nij", extra_odf, VV)
    O4_in = np.einsum("nv,vijkl->nijkl", intra_odf, VV4)
    O4_ex = np.einsum("nv,vijkl->nijkl", extra_odf, VV4)
    w_in = intra_odf.sum(1)                              # total intra weight
    w_ex = extra_odf.sum(1)
    _check_mixture_weights(w_in + w_ex + gm_frac + csf_frac + soma_frac)

    # --- mean tensor D_app -------------------------------------------------
    a = d_par[:, None, None]
    p_i = ip[:, None, None]
    p_e = d_perp[:, None, None]
    D_in = p_i * (w_in[:, None, None] * I3) + (a - p_i) * Om_in
    D_ex = p_e * (w_ex[:, None, None] * I3) + (a - p_e) * Om_ex
    D_gm = (gm_frac * gm_d)[:, None, None] * I3
    D_csf = (csf_frac * csf_d)[:, None, None] * I3
    D_soma = (soma_frac * soma_d)[:, None, None] * I3
    D_app = D_in + D_ex + D_gm + D_csf + D_soma          # (n,3,3)

    # --- second moment <D (x) D> ------------------------------------------
    def _second_moment(w, Om, O4, dpar, dperp):
        q = (dpar - dperp)[:, None, None, None, None]
        dp = dperp[:, None, None, None, None]
        IxOm = np.einsum("ij,nkl->nijkl", I3, Om)
        OmxI = np.einsum("nij,kl->nijkl", Om, I3)
        return (dp ** 2 * w[:, None, None, None, None] * IxI
                + dp * q * (IxOm + OmxI)
                + q ** 2 * O4)

    DD = (_second_moment(w_in, Om_in, O4_in, d_par, ip)
          + _second_moment(w_ex, Om_ex, O4_ex, d_par, d_perp)
          + (gm_frac * gm_d ** 2)[:, None, None, None, None] * IxI
          + (csf_frac * csf_d ** 2)[:, None, None, None, None] * IxI
          + (soma_frac * soma_d ** 2)[:, None, None, None, None] * IxI)
    C = DD - np.einsum("nij,nkl->nijkl", D_app, D_app)
    return D_app, C


def synth_signal_from_odfs(
    intra_odf, extra_odf, verts, d_par, d_perp,
    gm_frac, gm_d, csf_frac, csf_d, bvals, bvecs,
    *, soma_frac=0.0, soma_d=1.0e-3,
):
    """Noise-free mixture signal (S0=1) on an arbitrary gradient table.

    Used to build a *canonical* scheme-independent signal for the
    ``metric_method='canonical'`` DTI/DKI fit and for the anisotropic MAP-MRI
    fit that yields NG/PA.  ``soma_frac``/``soma_d`` add the optional isotropic
    soma/dot compartment.
    """
    intra_odf = np.atleast_2d(np.asarray(intra_odf, np.float64))
    extra_odf = np.atleast_2d(np.asarray(extra_odf, np.float64))
    verts = np.asarray(verts, np.float64)
    bvals = np.asarray(bvals, np.float64)
    bvecs = np.asarray(bvecs, np.float64)
    n = intra_odf.shape[0]
    G = bvals.shape[0]
    d_par = np.broadcast_to(np.asarray(d_par, np.float64), (n,))
    d_perp = np.broadcast_to(np.asarray(d_perp, np.float64), (n,))
    gm_frac = np.broadcast_to(np.asarray(gm_frac, np.float64), (n,))
    gm_d = np.broadcast_to(np.asarray(gm_d, np.float64), (n,))
    csf_frac = np.broadcast_to(np.asarray(csf_frac, np.float64), (n,))
    csf_d = np.broadcast_to(np.asarray(csf_d, np.float64), (n,))
    soma_frac = np.broadcast_to(np.asarray(soma_frac, np.float64), (n,))
    soma_d = np.broadcast_to(np.asarray(soma_d, np.float64), (n,))

    cos2 = (bvecs @ verts.T) ** 2                        # (G, M) = (g.v)^2
    bb = bvals[None, :, None]                            # (1,G,1)
    cc = cos2[None, :, :]                                # (1,G,M)
    S = np.empty((n, G), np.float64)
    # chunk voxels to bound the (chunk, G, M) intermediate
    step = max(1, int(4e7 // max(G * cos2.shape[1], 1)))
    for s in range(0, n, step):
        e = min(s + step, n)
        dpar = d_par[s:e, None, None]
        dperp = d_perp[s:e, None, None]
        exp_in = np.exp(-bb * dpar * cc)                 # (k,G,M) stick
        exp_ex = np.exp(-bb * (dperp + (dpar - dperp) * cc))  # (k,G,M) zeppelin
        Sk = (np.einsum("nv,ngv->ng", intra_odf[s:e], exp_in)
              + np.einsum("nv,ngv->ng", extra_odf[s:e], exp_ex))
        Sk += gm_frac[s:e, None] * np.exp(-bvals[None, :] * gm_d[s:e, None])
        Sk += csf_frac[s:e, None] * np.exp(-bvals[None, :] * csf_d[s:e, None])
        Sk += soma_frac[s:e, None] * np.exp(-bvals[None, :] * soma_d[s:e, None])
        S[s:e] = Sk
    return S


# ---------------------------------------------------------------------------
# closed-form MAP-MRI q-space indices (validated ratio 1.0000 vs MapmriFit)
# ---------------------------------------------------------------------------
def mapmri_closed_form_indices(
    intra_odf, extra_odf, verts, d_par, d_perp,
    gm_frac, gm_d, csf_frac, csf_d, tau, *, soma_frac=0.0, soma_d=1.0e-3,
    d_perp_floor=0.12e-3,
):
    """RTOP, RTAP, RTPP, MSD, QIV in closed form from the Gaussian mixture.

    A small intra-axonal radial-diffusivity floor ``d_perp_floor`` regularizes
    the otherwise-singular stick compartments for RTOP/RTAP/RTPP/QIV; it does
    not affect the stored signal.  Returns a dict of length-N arrays.

    Units caveat: ``tau`` sets the physical scale.  When the gradient table
    lacks ``big_delta``/``small_delta`` the caller passes the dipy default
    ``tau = 1/(4*pi**2)``, in which case the absolute index values are in
    *normalized* (dimensionless) units -- only voxel-to-voxel contrast is
    meaningful.  Supply diffusion timings for physical (mm-based) values.
    Assumes the compartment weights sum to 1 per voxel.
    """
    intra_odf = np.atleast_2d(np.asarray(intra_odf, np.float64))
    extra_odf = np.atleast_2d(np.asarray(extra_odf, np.float64))
    verts = np.asarray(verts, np.float64)
    n = intra_odf.shape[0]
    d_par = np.broadcast_to(np.asarray(d_par, np.float64), (n,))
    d_perp = np.broadcast_to(np.asarray(d_perp, np.float64), (n,))
    gm_frac = np.broadcast_to(np.asarray(gm_frac, np.float64), (n,))
    gm_d = np.broadcast_to(np.asarray(gm_d, np.float64), (n,))
    csf_frac = np.broadcast_to(np.asarray(csf_frac, np.float64), (n,))
    csf_d = np.broadcast_to(np.asarray(csf_d, np.float64), (n,))
    soma_frac = np.broadcast_to(np.asarray(soma_frac, np.float64), (n,))
    soma_d = np.broadcast_to(np.asarray(soma_d, np.float64), (n,))
    fl = float(d_perp_floor)

    w_in = intra_odf.sum(1)
    w_ex = extra_odf.sum(1)

    # mean tensor for the axis and for MSD (use TRUE sticks: intra d_perp=0)
    D_app, _ = moments_from_odfs(
        intra_odf, extra_odf, verts, d_par, d_perp,
        gm_frac, gm_d, csf_frac, csf_d,
        soma_frac=soma_frac, soma_d=soma_d, intra_dperp=0.0,
    )
    axis = np.linalg.eigh(D_app)[1][:, :, -1]            # principal evec (n,3)

    four_pi_tau = 4.0 * np.pi * tau

    # --- MSD (true tensors) ------------------------------------------------
    msd = 2.0 * tau * np.trace(D_app, axis1=1, axis2=2)

    # --- rotation-invariant reductions (floored intra) --------------------
    det_in = d_par * fl ** 2
    det_ex = d_par * d_perp ** 2
    rtop = four_pi_tau ** -1.5 * (
        w_in * det_in ** -0.5 + w_ex * det_ex ** -0.5
        + gm_frac * gm_d ** -1.5 + csf_frac * csf_d ** -1.5
        + soma_frac * soma_d ** -1.5
    )
    # QIV = 4 pi^2 / (-nabla^2 P(0)); nabla^2 N_c(0) = -N_c(0) trace(Sigma_c^-1)
    def _lap_iso(w, d):     # isotropic compartment, Sigma=2 tau d I
        s = 2.0 * tau * d
        N0 = (2 * np.pi * s) ** -1.5
        return w * N0 * (3.0 / s)
    def _lap_aniso(w, a, p):   # eigenvalues (a,p,p), Sigma=2 tau diag
        sa, sp = 2 * tau * a, 2 * tau * p
        N0 = (2 * np.pi) ** -1.5 * (sa * sp * sp) ** -0.5
        return w * N0 * (1.0 / sa + 2.0 / sp)
    lap = -(
        _lap_aniso(w_in, d_par, fl) + _lap_aniso(w_ex, d_par, d_perp)
        + _lap_iso(gm_frac, gm_d) + _lap_iso(csf_frac, csf_d)
        + _lap_iso(soma_frac, soma_d)
    )
    qiv = -4.0 * np.pi ** 2 / lap

    # --- axis-dependent reductions (floored intra) -----------------------
    cos2 = (verts @ axis.T).T ** 2                        # (n, M) = (v . a_n)^2
    sin2 = 1.0 - cos2
    # parallel diffusivity along the axis: a^T D a = a_par cos^2 + a_perp sin^2
    dpar_in = d_par[:, None] * cos2 + fl * sin2
    dpar_ex = d_par[:, None] * cos2 + d_perp[:, None] * sin2
    rtpp = four_pi_tau ** -0.5 * (
        (intra_odf / np.sqrt(dpar_in)).sum(1)
        + (extra_odf / np.sqrt(dpar_ex)).sum(1)
        + gm_frac / np.sqrt(gm_d) + csf_frac / np.sqrt(csf_d)
        + soma_frac / np.sqrt(soma_d)
    )
    # perpendicular 2x2 determinant of an axisymmetric tensor (evals a,p,p)
    # whose axis makes angle theta with a_hat (cos^2 = cos2):
    #   det(D^perp) = p * (a sin^2 + p cos^2)
    detperp_in = fl * (d_par[:, None] * sin2 + fl * cos2)
    detperp_ex = d_perp[:, None] * (d_par[:, None] * sin2 + d_perp[:, None] * cos2)
    rtap = four_pi_tau ** -1.0 * (
        (intra_odf / np.sqrt(detperp_in)).sum(1)
        + (extra_odf / np.sqrt(detperp_ex)).sum(1)
        + gm_frac / gm_d + csf_frac / csf_d
        + soma_frac / soma_d
    )
    return {"rtop": rtop, "rtap": rtap, "rtpp": rtpp, "msd": msd, "qiv": qiv}


# MAP-MRI / 3D-SHORE index keys produced by each fit model.
MAPMRI_FIT_KEYS = ("rtop", "rtap", "rtpp", "msd", "qiv", "ng", "ngpar", "ngperp")
SHORE_FIT_KEYS = ("rtop", "msd")


def mapmri_indices_via_fit(signals, gtab, *, model="mapmri", radial_order=6,
                           laplacian_weighting="GCV"):
    """MAP-MRI / 3D-SHORE q-space indices from a *fit* to ``signals``.

    An alternative to :func:`mapmri_closed_form_indices`: fit dipy's
    ``MapmriModel`` (``model='mapmri'``) or ``ShoreModel`` (``model='shore'``)
    to the (already S0-normalized) signals and read the indices from the fitted
    coefficients.  When the signals are synthesized on a fixed canonical
    q-space scheme (see ``synth_signal_from_odfs``) this is scheme-independent
    and lets the closed-form indices be benchmarked against a real fit; the
    MAP-MRI fit additionally yields NG / NGpar / NGperp.

    Returns a dict of length-N arrays.  MAP-MRI keys: rtop, rtap, rtpp, msd,
    qiv, ng, ngpar, ngperp.  SHORE keys: rtop, msd.
    """
    from dipy.reconst import mapmri as _mm, shore as _sh

    signals = np.atleast_2d(np.asarray(signals, np.float64))
    if model == "mapmri":
        fit = _mm.MapmriModel(
            gtab, radial_order=radial_order, laplacian_regularization=True,
            laplacian_weighting=laplacian_weighting,
        ).fit(signals)
        return {
            "rtop": np.asarray(fit.rtop()), "rtap": np.asarray(fit.rtap()),
            "rtpp": np.asarray(fit.rtpp()), "msd": np.asarray(fit.msd()),
            "qiv": np.asarray(fit.qiv()), "ng": np.asarray(fit.ng()),
            "ngpar": np.asarray(fit.ng_parallel()),
            "ngperp": np.asarray(fit.ng_perpendicular()),
        }
    elif model == "shore":
        fit = _sh.ShoreModel(gtab, radial_order=radial_order).fit(signals)
        return {"rtop": np.asarray(fit.rtop_signal()),
                "msd": np.asarray(fit.msd())}
    raise ValueError(f"model must be 'mapmri' or 'shore', got {model!r}")


# ---------------------------------------------------------------------------
# closed-form non-Gaussianity (NG) and propagator anisotropy (PA)
# ---------------------------------------------------------------------------
def _isotropic_scale(evals):
    """Ozarslan isotropic scale (in diffusivity units): the largest positive
    real root of the cubic that best matches an anisotropic scale by an
    isotropic one (:footcite:p:`Ozarslan2013` eq. 49)."""
    X, Y, Z = evals
    roots = np.roots([-3.0, -(X + Y + Z), X * Y + X * Z + Y * Z, 3.0 * X * Y * Z])
    roots = roots[np.abs(roots.imag) < 1e-9].real
    roots = roots[roots > 0]
    return float(roots.max()) if roots.size else float(np.mean(evals))


def gaussian_mixture_ng_pa(weights, tensors, *, floor=0.12e-3, axis=None):
    """Closed-form non-Gaussianity and propagator anisotropy of a Gaussian
    mixture ``P = sum_c w_c N(0, 2 tau D_c)``.

    Every quantity is a dimensionless angle in propagator space (tau cancels),
    computed from Gaussian overlaps ``<N(0,A),N(0,B)> ~ det(A+B)**-1/2``.

    Returns a dict with:

    * ``ng``     -- sine of the angle between the propagator and its best-fit
      Gaussian (the DTI propagator); 0 for a single Gaussian, grows with
      restriction / compartment heterogeneity (a normalized twin of kurtosis).
    * ``ngpar`` / ``ngperp`` -- the same for the 1D axial and 2D perpendicular
      marginals; ``ngperp`` isolates perpendicular restriction.
    * ``pa``     -- anisotropy of the mean-tensor Gaussian relative to its
      best isotropic match; 0 for isotropic tissue, grows with directional
      coherence (a propagator-space anisotropy, monotone with FA).

    A small ``floor`` regularizes otherwise-singular stick compartments.
    """
    from dipy.sims.voxel import all_tensor_evecs

    w = np.asarray(weights, np.float64)
    w = w / w.sum()
    tw, tv = np.linalg.eigh(np.asarray(tensors, np.float64))
    tw = np.clip(tw, floor, None)
    T = (tv * tw[..., None, :]) @ np.swapaxes(tv, -1, -2)
    D = np.einsum("c,cij->ij", w, T)

    def _ng(mats, ref):        # generic mixture-vs-best-Gaussian angle
        ovpp = np.sum(np.outer(w, w) * np.linalg.det(mats[:, None] + mats[None, :]) ** -0.5)
        ovp0 = np.sum(w * np.linalg.det(mats + ref) ** -0.5)
        ov00 = np.linalg.det(2 * ref) ** -0.5
        return np.sqrt(max(0.0, 1.0 - ovp0 ** 2 / (ovpp * ov00))) if ovpp > 0 else 0.0

    ng = _ng(T, D)
    ev, evec = np.linalg.eigh(D)
    axis = evec[:, -1] if axis is None else np.asarray(axis, np.float64)

    # 1D axial marginal: variances d_par along the axis
    dpar = np.einsum("i,cij,j->c", axis, T, axis)
    da = float(axis @ D @ axis)
    ovpp = np.sum(np.outer(w, w) * (dpar[:, None] + dpar[None, :]) ** -0.5)
    ngpar = np.sqrt(max(0.0, 1.0 - np.sum(w * (dpar + da) ** -0.5) ** 2
                        / (ovpp * (2 * da) ** -0.5)))

    # 2D perpendicular marginal
    Pb = all_tensor_evecs(axis)[:, 1:]
    Tp = np.einsum("ia,cij,jb->cab", Pb, T, Pb)
    Ta = Pb.T @ D @ Pb
    ngperp = _ng(Tp, Ta)

    # PA: anisotropy of the mean-tensor Gaussian vs its best isotropic match
    sI = _isotropic_scale(ev) * np.eye(3)
    cos2 = np.linalg.det(D + sI) ** -1.0 / (
        np.linalg.det(2 * D) ** -0.5 * np.linalg.det(2 * sI) ** -0.5
    )
    pa = np.sqrt(max(0.0, 1.0 - cos2))
    return {"ng": ng, "ngpar": ngpar, "ngperp": ngperp, "pa": pa}


NG_PA_KEYS = ("ng", "ngpar", "ngperp", "pa")


def ng_pa_from_odfs(
    intra_odf, extra_odf, verts, d_par, d_perp, gm_frac, gm_d, csf_frac, csf_d,
    *, soma_frac=0.0, soma_d=1.0e-3, floor=0.12e-3, top_k=28,
):
    """Batched closed-form NG/NGpar/NGperp/PA for FORCE library entries,
    reconstructing each entry's Gaussian mixture from the emitted orientation
    weights.  Returns a dict of length-N arrays.  ``top_k`` keeps only the
    strongest orientation-weight directions per entry for efficiency.
    ``soma_frac``/``soma_d`` add the optional isotropic soma/dot compartment."""
    intra_odf = np.atleast_2d(np.asarray(intra_odf, np.float64))
    extra_odf = np.atleast_2d(np.asarray(extra_odf, np.float64))
    verts = np.asarray(verts, np.float64)
    n = intra_odf.shape[0]

    def _b(x):
        return np.broadcast_to(np.asarray(x, np.float64), (n,))

    d_par, d_perp = _b(d_par), _b(d_perp)
    gm_frac, gm_d = _b(gm_frac), _b(gm_d)
    csf_frac, csf_d = _b(csf_frac), _b(csf_d)
    soma_frac, soma_d = _b(soma_frac), _b(soma_d)
    out = {k: np.zeros(n, np.float64) for k in NG_PA_KEYS}
    I3 = np.eye(3)
    for i in range(n):
        tot = intra_odf[i] + extra_odf[i]
        sel = np.argsort(tot)[::-1][:top_k]
        sel = sel[tot[sel] > 0]
        v = verts[sel]
        vv = np.einsum("vi,vj->vij", v, v)
        Tin = d_par[i] * vv + floor * (I3 - vv)
        Tex = d_perp[i] * I3 + (d_par[i] - d_perp[i]) * vv
        w = np.concatenate([intra_odf[i][sel], extra_odf[i][sel],
                            [gm_frac[i]], [csf_frac[i]], [soma_frac[i]]])
        T = np.concatenate([Tin, Tex, (gm_d[i] * I3)[None], (csf_d[i] * I3)[None],
                            (soma_d[i] * I3)[None]])
        m = w > 1e-6
        r = gaussian_mixture_ng_pa(w[m], T[m], floor=floor)
        for k in NG_PA_KEYS:
            out[k][i] = r[k]
    return out


# ---------------------------------------------------------------------------
# QTI / DIVIDE invariants (Westin 2016) -- closed form from the covariance
# ---------------------------------------------------------------------------
# Voigt index map matching qti.from_3x3_to_6x1: [00, 11, 22, s12, s02, s01]
_VOIGT_I = np.array([0, 1, 2, 1, 0, 0])
_VOIGT_J = np.array([0, 1, 2, 2, 2, 1])
_VOIGT_S = np.array([1.0, 1.0, 1.0, np.sqrt(2.0), np.sqrt(2.0), np.sqrt(2.0)])

QTI_KEYS = ("micro_fa", "coherence", "k_bulk", "k_shear")


def _cov_rank4_to_voigt_6x6(C):
    """Rank-4 Cartesian covariance ``(...,3,3,3,3)`` -> 6x6 Voigt ``(...,6,6)``.

    Uses the same sqrt(2) off-diagonal convention as ``qti.from_3x3_to_6x1`` so
    the result is the QTI covariance tensor of the diffusion-tensor
    distribution.
    """
    Ia, Ja = _VOIGT_I[:, None], _VOIGT_J[:, None]
    Ib, Jb = _VOIGT_I[None, :], _VOIGT_J[None, :]
    ss = _VOIGT_S[:, None] * _VOIGT_S[None, :]
    return ss * C[..., Ia, Ja, Ib, Jb]


def qti_indices_from_moments(D_app, C):
    """Closed-form QTI/DIVIDE invariants from the mixture mean tensor + covariance.

    The FORCE covariance ``C`` *is* the diffusion-tensor-distribution covariance
    of :footcite:t:`Westin2016`, so the tensor-valued-encoding (DDE) invariants
    are available in closed form -- FORCE reports microscopic anisotropy and the
    isotropic/anisotropic kurtosis split *without* needing tensor-valued
    acquisitions.  Reuses :class:`dipy.reconst.qti.QtiFit` (unmodified).

    Returns a dict of length-N arrays:

    * ``micro_fa``  -- microscopic FA (uFA); the per-compartment shape
      anisotropy, invariant to orientation dispersion.
    * ``coherence`` -- microscopic orientation coherence ``C_c = C_M / C_mu``;
      ~1 for a single coherent fibre, <1 for crossing/dispersion.
    * ``k_bulk``    -- isotropic (size-variance) kurtosis; the free-water /
      compartment-size-heterogeneity part of MK.
    * ``k_shear``   -- anisotropic (shape-variance) kurtosis; the fibre part.
    """
    from dipy.reconst import qti

    D_app = np.asarray(D_app, np.float64)
    C = np.asarray(C, np.float64)
    batch = D_app.shape[:-2]
    Dm = D_app.reshape(-1, 3, 3)
    Cm = C.reshape((-1,) + C.shape[-4:])
    n = Dm.shape[0]

    D6 = qti.from_3x3_to_6x1(Dm)[..., 0]                      # (n, 6)
    C21 = qti.from_6x6_to_21x1(_cov_rank4_to_voigt_6x6(Cm))[..., 0]  # (n, 21)
    params = np.concatenate([np.ones((n, 1)), D6, C21], axis=1)      # (n, 28)
    fit = qti.QtiFit(params)
    # micro_fa = sqrt(c_mu) and coherence = c_m/c_mu are ill-defined for a
    # purely isotropic mixture (c_mu = 0); clip/guard instead of NaN.
    c_mu = np.asarray(fit.c_mu, np.float64)
    c_m = np.asarray(fit.c_m, np.float64)
    micro_fa = np.sqrt(np.clip(c_mu, 0.0, None))
    coherence = np.where(
        c_mu > 1e-12, np.clip(c_m / np.where(c_mu > 1e-12, c_mu, 1.0), 0.0, 1.0), 0.0
    )
    out = {
        "micro_fa": micro_fa,
        "coherence": coherence,
        "k_bulk": np.asarray(fit.k_bulk, np.float64),
        "k_shear": np.asarray(fit.k_shear, np.float64),
    }
    return {k: np.asarray(v, np.float64).reshape(batch) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Restriction Spectrum Imaging (ABCD RSIproc convention) -- N0/ND/NT
# ---------------------------------------------------------------------------
# This is a faithful port of the ABCD-STUDY ``RSIproc`` operator (White 2013;
# Hagler 2019) so that the FORCE-stored RSI columns match the reference pipeline
# *numerically*, not merely in spirit.  Each compartment ``c`` is a fixed
# (D_long, D_trans, SH_order) response; the design projects the signal onto a
# real-SH-convolution basis and the measures are Euclidean norms of the fitted
# coefficient blocks:
#
#   N0[c] = beta(l=0)          (isotropic component; signed, clipped [-1, 2])
#   ND[c] = ||beta(l>0)||      (directional component; 0 for SH_order 0)
#   NT[c] = ||beta(all l)||    (total component)   ->  NT**2 = N0**2 + ND**2
#
# Defaults reproduce RSIproc_1_0_8.py exactly: restricted stick (DT->0, l<=4),
# hindered zeppelin (DT=0.9e-3, l<=4), free isotropic (l=0); uniform Tikhonov
# regularization ``lambda=0.1 * mean(diag(A^T A))``; ``normalize=False`` (raw
# betas, RSIproc default).  Override ``compartments``/``rsi_lambda``/``normalize``
# to mirror a differently-configured RSI pipeline.
#
# (name, D_longitudinal, D_transverse, SH_order) in mm^2/s.
RSI_COMPARTMENTS = (
    ("r", 1.0e-3, 1.0e-10, 4),   # restricted stick (DT->0 == RSIproc 1e-10)
    ("h", 1.0e-3, 0.9e-3, 4),    # hindered zeppelin
    ("f", 3.0e-3, 3.0e-3, 0),    # free water (isotropic, l=0)
)
RSI_LAMBDA = 0.1                 # RSIproc RSI_lambda (uniform regularization)
RSI_ICO_ORDER = 3                # RSIproc make_icosahedron(3) -> 642 vertices
RSI_KEYS = ("rnt", "rn0", "rnd", "hnt", "hn0", "hnd", "fnt")


def _rsi_icosahedron(order=RSI_ICO_ORDER):
    """Unit-sphere vertices of a subdivided icosahedron, reproducing the
    reconstruction sphere of ABCD ``RSIproc`` (``make_icosahedron(order)``);
    ``order=3`` yields 642 points.  The RSI measures are norms of SH-coefficient
    blocks, so the *vertex order is irrelevant* -- only the point set (hence its
    density, which sets the coefficient scale) matters, and this reproduces it
    exactly."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
    ]
    verts = [np.asarray(v, np.float64) / np.linalg.norm(v) for v in verts]
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]

    def _midpoint(cache, a, b):
        key = (a, b) if a < b else (b, a)
        if key not in cache:
            m = (verts[a] + verts[b]) * 0.5
            verts.append(m / np.linalg.norm(m))
            cache[key] = len(verts) - 1
        return cache[key]

    for _ in range(order):
        cache, new_faces = {}, []
        for a, b, c in faces:
            ab = _midpoint(cache, a, b)
            bc = _midpoint(cache, b, c)
            ca = _midpoint(cache, c, a)
            new_faces += [(a, ab, ca), (ab, b, bc), (bc, c, ca), (ab, bc, ca)]
        faces = new_faces
    return np.asarray(verts)


def rsi_operator(bvals, bvecs, *, compartments=RSI_COMPARTMENTS,
                 rsi_lambda=RSI_LAMBDA, ico_order=RSI_ICO_ORDER):
    """Build the ABCD RSIproc linear estimator ``W`` (and its column layout) for
    a gradient scheme.  ``beta = W @ (S / S0)`` recovers the RSI coefficients.

    Returns ``(W, L, cid)`` where ``W`` is ``(nFit, nMeas)``, ``L`` the SH order
    per coefficient, and ``cid`` the compartment id per coefficient.  Faithful to
    ``RSIproc.calculateA``/``calculateW``: ``A = (R @ pinv(Y_l).T) * Y00`` per
    compartment, ``W = (A^T A + lambda*mean(diag(A^T A)) I)^-1 A^T``."""
    from dipy.core.geometry import cart2sphere
    from dipy.reconst.shm import real_sh_descoteaux_from_index, sph_harm_ind_list

    bvals = np.asarray(bvals, np.float64)
    bvecs = np.asarray(bvecs, np.float64)
    verts = _rsi_icosahedron(ico_order)
    max_order = max(c[3] for c in compartments)
    m_l, l_l = sph_harm_ind_list(max_order)
    _, theta, phi = cart2sphere(verts[:, 0], verts[:, 1], verts[:, 2])
    YL = np.stack([real_sh_descoteaux_from_index(m, l, theta, phi)
                   for m, l in zip(m_l, l_l)], axis=1)          # (M, nSHmax)
    Y00 = float(real_sh_descoteaux_from_index(0, 0, theta[:1], phi[:1])[0])
    cos2 = (bvecs @ verts.T) ** 2                              # (nMeas, M)

    blocks, Ls, cids = [], [], []
    for ci, (_, DL, DT, sho) in enumerate(compartments):
        sel = l_l <= sho                                       # contiguous l-prefix
        R = np.exp(-bvals[:, None] * (DT + (DL - DT) * cos2))  # (nMeas, M)
        A = R @ np.linalg.pinv(YL[:, sel]).T                   # (nMeas, nSH_c)
        blocks.append(A)
        Ls.append(l_l[sel])
        cids.append(np.full(int(sel.sum()), ci))
    A = np.hstack(blocks) * Y00
    L = np.concatenate(Ls)
    cid = np.concatenate(cids)
    AtA = A.T @ A
    W = np.linalg.solve(
        AtA + rsi_lambda * np.mean(np.diag(AtA)) * np.eye(A.shape[1]), A.T)
    return W, L, cid


def rsi_indices_from_signals(signals, gtab, verts=None, *,
                             compartments=RSI_COMPARTMENTS,
                             rsi_lambda=RSI_LAMBDA, normalize=False,
                             ico_order=RSI_ICO_ORDER):
    """ABCD-RSIproc directional RSI measures (N0/ND/NT per compartment) for a set
    of FORCE library signals, by the *exact fixed-basis linear projection* of the
    signals through the RSIproc operator -- i.e. the noise-free limit of an RSI
    fit, with no per-voxel iterative fitting.

    ``signals`` : (n, n_meas) signals; ``gtab`` : their gradient table.  ``verts``
    is accepted for backwards compatibility but ignored -- RSI uses its own fixed
    icosahedral reconstruction sphere (``ico_order``).  With ``normalize=False``
    (the RSIproc default) the returned N0/ND/NT are raw coefficient norms that
    match the reference pipeline numerically; with ``normalize=True`` each is
    divided by the total coefficient L2-norm so ``NT**2`` is a signal fraction.

    Returns ``rnt/rn0/rnd`` (restricted), ``hnt/hn0/hnd`` (hindered), ``fnt``
    (free); per compartment ``NT**2 = N0**2 + ND**2``.
    """
    signals = np.atleast_2d(np.asarray(signals, np.float64))
    bvals = np.asarray(gtab.bvals, np.float64)
    W, L, cid = rsi_operator(bvals, np.asarray(gtab.bvecs, np.float64),
                             compartments=compartments, rsi_lambda=rsi_lambda,
                             ico_order=ico_order)
    b0 = bvals <= getattr(gtab, "b0_threshold", 50)
    S0 = signals[:, b0].mean(1) if np.any(b0) else signals.max(1)
    beta = (signals / np.maximum(S0, 1e-9)[:, None]) @ W.T     # (n, nFit)
    full_norm = (np.linalg.norm(beta, axis=1) if normalize
                 else np.ones(len(beta)))
    full_norm = np.maximum(full_norm, 1e-12)

    out = {}
    for ci, (name, _DL, _DT, _sho) in enumerate(compartments):
        b = beta[:, cid == ci]
        li = L[cid == ci]
        n0 = b[:, li == 0][:, 0] / full_norm                   # signed (RSIproc)
        if not normalize:                                      # RSIproc n0 clip
            n0 = np.clip(n0, -1.0, 2.0)
        nt = np.linalg.norm(b, axis=1) / full_norm
        nd = (np.linalg.norm(b[:, li > 0], axis=1) / full_norm
              if np.any(li > 0) else np.zeros(len(b)))
        out[name] = (n0, nd, nt)

    r, h, f = out["r"], out["h"], out["f"]
    return {
        "rnt": r[2], "rn0": r[0], "rnd": r[1],
        "hnt": h[2], "hn0": h[0], "hnd": h[1],
        "fnt": f[2],
    }


# ---------------------------------------------------------------------------
# RISH -- rotation-invariant spherical harmonic power features
# ---------------------------------------------------------------------------
# For each non-b0 shell, ||c_l|| = sqrt(sum_m c_lm^2) for l = 0, 2, ... These are
# exactly the rotation-invariant content of the signal: l=0 is the spherical mean
# (bulk microstructure) while l>=2 carries dispersion / crossing structure *with
# no orientation*.  Matching a library on RISH therefore decouples the
# microstructure search from orientation -- FORCE's raw-signal cosine match is
# ~90% orientation-blind anyway (most of its L2 energy is b0 + spherical mean),
# so it wastes library capacity on rotated copies while still failing to pin
# orientation.  RISH keeps the useful half and drops the nuisance.
#
# Basis: MRtrix3 (``tournier07``, ``legacy=False``).  NOTE the per-l power is
# invariant to *any* orthonormal real-SH basis (descoteaux and tournier differ by
# a within-l orthogonal transform), so these features are basis-agnostic; we use
# the MRtrix3 convention for consistency with exported SH.
RISH_SH_ORDER = 6


def _even_sh_order_for(n_dirs, sh_order_max):
    """Largest even SH order whose basis is not underdetermined by n_dirs."""
    L = int(sh_order_max)
    while L > 0 and (L + 1) * (L + 2) // 2 > n_dirs:
        L -= 2
    return max(L, 0)


def rish_features_from_signals(signals, gtab, *, sh_order_max=RISH_SH_ORDER,
                               shell_tol=100.0, smooth=0.006, legacy=False):
    """Per-shell rotation-invariant SH power (RISH) features.

    The SH fit is Laplace-Beltrami regularized (``smooth``, dipy's CSD default),
    and the SH order is reduced per shell to keep the fit over-determined
    (``n_dirs >= 2 * n_coef``).  Both matter: RISH is *exactly* rotation invariant
    only in the continuous limit -- with a finite gradient set the fit residual
    depends on how the signal lands on the sampling grid, and an under-regularized
    high-l fit turns that into several percent of spurious rotational variance.

    Parameters
    ----------
    signals : ndarray (n, n_meas)
        Signals (any scale; normalized internally by the b0 mean).
    gtab : GradientTable
    sh_order_max : int, optional
        Maximum even SH order; reduced per shell if a shell is too sparse.
    shell_tol : float, optional
        b-value rounding used to group volumes into shells.
    smooth : float, optional
        Laplace-Beltrami regularization weight.

    Returns
    -------
    dict
        ``{"rish_b<shell>_l<l>": (n,) ndarray}``, dimensionless (S0-normalized).
    """
    from dipy.core.geometry import cart2sphere
    from dipy.reconst.shm import real_sh_tournier_from_index, sph_harm_ind_list

    signals = np.atleast_2d(np.asarray(signals, np.float64))
    bvals = np.asarray(gtab.bvals, np.float64)
    bvecs = np.asarray(gtab.bvecs, np.float64)

    b0 = bvals <= getattr(gtab, "b0_threshold", 50)
    S0 = signals[:, b0].mean(1) if np.any(b0) else signals.max(1)
    S = signals / np.maximum(S0, 1e-12)[:, None]

    shells = np.round(bvals / shell_tol) * shell_tol
    out = {}
    for shell in np.unique(shells[~b0]):
        m = (shells == shell) & (~b0)
        n_dirs = int(m.sum())
        # keep the fit comfortably over-determined
        L = _even_sh_order_for(n_dirs // 2, sh_order_max)
        if L < 2:                                    # too sparse to be useful
            continue
        m_l, l_l = sph_harm_ind_list(L)
        _, theta, phi = cart2sphere(*bvecs[m].T)
        B = real_sh_tournier_from_index(m_l, l_l, theta[:, None], phi[:, None],
                                        legacy=legacy)
        # Laplace-Beltrami regularized least squares
        lb = (l_l ** 2) * ((l_l + 1) ** 2)
        BtB = B.T @ B
        coef = np.linalg.solve(BtB + smooth * np.diag(lb), B.T @ S[:, m].T).T
        for L_i in range(0, L + 1, 2):
            out[f"rish_b{int(shell)}_l{L_i}"] = np.linalg.norm(
                coef[:, l_l == L_i], axis=1)
    return out


# ---------------------------------------------------------------------------
# Generalized q-sampling imaging (GQI) ODF-shape statistics -- GFA / QA
# ---------------------------------------------------------------------------
GQI_KEYS = ("gfa", "qa")


def gqi_indices_from_odfs(odfs, wm_fraction):
    """Analytic GQI-style ODF-shape statistics (GFA, QA) for FORCE entries.

    The stored ``odfs`` are the (unnormalized) WM fibre ODF.  To make GFA/QA
    consistent with the other mixture statistics -- diluted by free water like
    FA/MD -- the fibre ODF is renormalized to a probability and mixed with an
    isotropic baseline weighted by the non-WM (GM+CSF) fraction::

        odf = wm_fraction * (fodf / sum fodf) + (1 - wm_fraction) / n_vertices

    **GFA** is dipy's generalized fractional anisotropy of that mixture ODF and
    **QA** its quantitative anisotropy ``max(odf) - min(odf)`` -- both functions
    of the ODF samples only (orientation- and scheme-independent).  Pure
    free-water entries (``wm_fraction -> 0``) give an isotropic ODF, so GFA/QA
    -> 0.

    Parameters
    ----------
    odfs : ndarray (n, n_vertices)
        Per-entry WM fibre ODF on the FORCE target sphere.
    wm_fraction : ndarray (n,)
        Per-entry white-matter signal fraction (dilutes the ODF anisotropy).

    Returns
    -------
    dict with ``gfa`` and ``qa`` arrays, each shape ``(n,)``.
    """
    from dipy.reconst.odf import gfa as _gfa

    odfs = np.atleast_2d(np.asarray(odfs, np.float64))
    wm = np.asarray(wm_fraction, np.float64).reshape(-1)
    n_vert = odfs.shape[1]
    ssum = odfs.sum(1, keepdims=True)
    fodf = np.divide(odfs, ssum, out=np.zeros_like(odfs), where=ssum > 0)
    odf = wm[:, None] * fodf + (1.0 - wm)[:, None] / n_vert
    gfa = np.atleast_1d(np.asarray(_gfa(odf))).reshape(-1)
    return {"gfa": gfa, "qa": odf.max(1) - odf.min(1)}
