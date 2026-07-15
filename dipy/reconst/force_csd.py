"""FORCE-informed multi-shell multi-tissue CSD.

FORCE is an excellent *kernel* (response-function) estimator but a poor
orientation estimator (its nearest-neighbour signal match is ~90% orientation-
blind).  Deconvolution is the right tool for orientation.  This module closes the
loop: a FORCE match hands over the exact per-voxel WM kernel
(``d_par``, ``d_perp``, ``f_intra``) and the isotropic diffusivities, from which
we build an **exact** multi-shell response and hand it to dipy's
``MultiShellDeconvModel``.  Unlike standard MSMT-CSD -- which uses a single
*global, single-tensor* WM response for the whole brain -- this response is
per-voxel and two-compartment (stick + zeppelin), so it should resolve crossings
the global response smears.

Cost: a deconvolution operator is expensive to build, so voxels are binned by
their kernel (k-means) and one ``MultiShellDeconvModel`` is built per bin.
"""
import numpy as np

from dipy.reconst import shm
from dipy.reconst.mcsd import MultiShellResponse, MultiShellDeconvModel
from dipy.data import default_sphere

SH_CONST = 0.5 / np.sqrt(np.pi)          # = real_sh_descoteaux(0,0,0,0), matches mcsd


def _wm_kernel_signal(bval, cos2, d_par, d_perp, f_intra):
    """Axially-symmetric WM response signal (fibre along +z) at one b-value.

    Stick + zeppelin: ``f_intra * exp(-b d_par cos^2) +
    (1-f_intra) * exp(-b (d_perp + (d_par-d_perp) cos^2))``.  ``f_intra=1`` with a
    zeppelin-style call reduces to a single tensor (used to validate against
    ``multi_shell_fiber_response``)."""
    stick = np.exp(-bval * d_par * cos2)
    zepp = np.exp(-bval * (d_perp + (d_par - d_perp) * cos2))
    return f_intra * stick + (1.0 - f_intra) * zepp


def force_response(shells, d_par, d_perp, f_intra, iso_ds, *, sh_order_max=8,
                   sphere=None):
    """Build a `MultiShellResponse` from a FORCE kernel.

    Parameters
    ----------
    shells : sequence of float
        Unique b-values *including* b0 (b0 first).
    d_par, d_perp, f_intra : float
        WM stick+zeppelin kernel parameters.
    iso_ds : sequence of float
        Isotropic-compartment diffusivities, one response column each (``iso``
        inferred from the count).  Order matches dipy's ``multi_shell_fiber_
        response`` convention -- the *slowest* (most CSF-like) first -- i.e.
        ``[csf_d, gm_d]`` for two compartments, or ``[csf_d, gm_d, soma_d]`` with
        soma.  The order only labels the returned isotropic volume fractions; the
        WM fODF and its peaks are unaffected.
    sh_order_max : int, optional

    Returns
    -------
    MultiShellResponse
        `response` has shape ``(n_shells, n_iso + n_even_l)`` matching dipy.
    """
    if sphere is None:
        sphere = default_sphere
    big = sphere.subdivide()
    theta, phi = big.theta, big.phi
    l_values = np.arange(0, sh_order_max + 1, 2)
    m_values = np.zeros_like(l_values)
    B = shm.real_sh_descoteaux_from_index(m_values, l_values,
                                          theta[:, None], phi[:, None])
    cos2 = np.cos(theta) ** 2                              # fibre along +z
    n_iso = len(iso_ds)
    shells = np.asarray(shells, float)
    response = np.empty((len(shells), n_iso + len(l_values)))
    for i, b in enumerate(shells):
        wm = _wm_kernel_signal(b, cos2, d_par, d_perp, f_intra)
        response[i, n_iso:] = np.linalg.lstsq(B, wm, rcond=None)[0]
        for j, d in enumerate(iso_ds):
            response[i, j] = np.exp(-b * d) / SH_CONST      # S0=1 convention
    return MultiShellResponse(response, sh_order_max, shells,
                              S0=np.ones(n_iso + 1))


def force_response_ssst(bval, d_par, d_perp, f_intra, *, sh_order_max=8,
                        sphere=None):
    """Single-shell FORCE response as an ``AxSymShResponse`` for
    ``ConstrainedSphericalDeconvModel`` (single-shell CSD).

    The stick+zeppelin kernel is evaluated at one b-value and projected onto the
    zonal (m=0) real-SH basis.  On a single low-b shell the two-compartment
    kernel is nearly a single tensor, so this response differs little from the
    standard tensor response -- which is the point: FORCE's advantage lives in
    the multi-shell / high-b structure a single shell cannot see.
    """
    from dipy.reconst.csdeconv import AxSymShResponse
    if sphere is None:
        sphere = default_sphere
    big = sphere.subdivide()
    theta = big.theta
    l_values = np.arange(0, sh_order_max + 1, 2)
    m_values = np.zeros_like(l_values)
    B = shm.real_sh_descoteaux_from_index(m_values, l_values,
                                          theta[:, None], big.phi[:, None])
    wm = _wm_kernel_signal(float(bval), np.cos(theta) ** 2,
                           d_par, d_perp, f_intra)          # S0=1 profile
    r_sh = np.linalg.lstsq(B, wm, rcond=None)[0]            # zonal SH coeffs
    return AxSymShResponse(1.0, r_sh)


def force_informed_fodf_ssst(signals, gtab, kernels, *, n_bins=150,
                             sh_order_max=8, sphere=None):
    """Single-shell FORCE-informed CSD (per-voxel response, binned).

    Returns ``(fodf_sh, labels)`` with descoteaux SH coefficients, matching
    ``force_informed_fodf`` so the two are drop-in interchangeable by shell count.
    """
    from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
    from dipy.core.gradients import unique_bvals_tolerance
    signals = np.asarray(signals, float)
    kernels = np.asarray(kernels, float)
    n = len(signals)
    shells = unique_bvals_tolerance(gtab.bvals, tol=50)
    bval = float(shells[shells > 50][0])                    # the single DW shell
    labels, centers = _bin_kernels(kernels[:, :3], n_bins)  # only d_par,d_perp,f_in
    n_sh = int((sh_order_max + 1) * (sh_order_max + 2) // 2)
    fodf = np.zeros((n, n_sh))
    for b in range(len(centers)):
        idx = np.where(labels == b)[0]
        if idx.size == 0:
            continue
        d_par, d_perp, f_in = centers[b]
        resp = force_response_ssst(bval, d_par, max(d_perp, 1e-5),
                                   float(np.clip(f_in, 0, 1)),
                                   sh_order_max=sh_order_max, sphere=sphere)
        model = ConstrainedSphericalDeconvModel(gtab, resp,
                                                sh_order_max=sh_order_max)
        fodf[idx] = model.fit(signals[idx]).shm_coeff
    return fodf, labels


def _bin_kernels(kernels, n_bins):
    """k-means the per-voxel kernels; return (labels, centroids)."""
    from sklearn.cluster import KMeans
    n_bins = int(min(n_bins, len(np.unique(kernels, axis=0))))
    km = KMeans(n_bins, n_init=3, random_state=0).fit(kernels)
    return km.labels_, km.cluster_centers_


def force_informed_fodf(signals, gtab, kernels, *, iso_names=("gm", "csf"),
                        n_bins=150, sh_order_max=8, sphere=None):
    """Deconvolve `signals` with per-voxel FORCE responses (binned for speed).

    Parameters
    ----------
    signals : ndarray (n, n_grad)
    gtab : GradientTable
    kernels : ndarray (n, 3 + n_iso)
        Per-voxel ``[d_par, d_perp, f_intra, *iso_ds]`` from the FORCE match.
    n_bins : int, optional
        Number of kernel bins (one deconvolution model per bin).

    Returns
    -------
    fodf_sh : ndarray (n, n_sh)   SH coefficients of the fODF (descoteaux basis).
    labels  : ndarray (n,)        kernel-bin assignment per voxel.
    """
    from dipy.core.gradients import unique_bvals_tolerance
    signals = np.asarray(signals, float)
    kernels = np.asarray(kernels, float)
    n = len(signals)
    shells = unique_bvals_tolerance(gtab.bvals, tol=50)
    labels, centers = _bin_kernels(kernels, n_bins)

    n_sh = int((sh_order_max + 1) * (sh_order_max + 2) // 2)
    fodf = np.zeros((n, n_sh))
    for b in range(len(centers)):
        idx = np.where(labels == b)[0]
        if idx.size == 0:
            continue
        d_par, d_perp, f_in = centers[b, :3]
        iso_ds = centers[b, 3:]
        resp = force_response(shells, d_par, max(d_perp, 1e-5),
                              float(np.clip(f_in, 0, 1)), iso_ds,
                              sh_order_max=sh_order_max, sphere=sphere)
        model = MultiShellDeconvModel(gtab, resp, sh_order_max=sh_order_max)
        fit = model.fit(signals[idx])
        fodf[idx] = fit.shm_coeff[..., 2:] if fit.shm_coeff.shape[-1] > n_sh \
            else fit.shm_coeff[..., -n_sh:]
    return fodf, labels
