"""Tests for analytic FORCE DTI/DKI/MAP-MRI scalars (dipy.sims._force_moments)
and the scheme-independent library generation in dipy.sims.force.

These lock in the fix for shell-based DTI/DKI computation failing on
non-shelled (cartesian / random / single-shell) acquisitions: the scalars are
now computed analytically from the known Gaussian mixture, independent of the
sampling scheme.
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest

from dipy.core.gradients import gradient_table
from dipy.data import default_sphere, get_sphere
from dipy.reconst import dki, dti, mapmri
from dipy.reconst.dki import apparent_kurtosis_coef
from dipy.sims._force_moments import (
    dki_params_from_moments,
    dki_params_from_tensor_distribution,
    gaussian_mixture_ng_pa,
    mapmri_closed_form_indices,
    moments_from_odfs,
    qti_indices_from_moments,
    synth_signal_from_odfs,
)
from dipy.sims.force import generate_force_simulations
from dipy.sims.voxel import all_tensor_evecs


def _rot_tensor(evals, u):
    R = all_tensor_evecs(np.asarray(u, float) / np.linalg.norm(u))
    return R @ np.diag(evals) @ R.T


def _multishell_gtab(shells, n_dir=64, seed=0, big_delta=0.043, small_delta=0.0106):
    rng = np.random.default_rng(seed)
    sph = get_sphere(name="repulsion724")
    dirs = sph.vertices[rng.choice(len(sph.vertices), n_dir, replace=False)]
    b = [0.0, 0.0]
    v = [[0, 0, 0], [0, 0, 0]]
    for s in shells:
        for d in dirs:
            b.append(float(s))
            v.append(d)
    return gradient_table(np.array(b), bvecs=np.array(v),
                          big_delta=big_delta, small_delta=small_delta)


def _crossing_mixture(seed=0):
    """Explicit Gaussian mixture: 2 dispersed crossing fibres + GM + CSF."""
    rng = np.random.default_rng(seed)
    d_par, d_perp = 2.3e-3, 0.55e-3
    weights, tensors = [], []
    for u, ff, fin in [(np.array([1.0, 0.1, 0.0]), 0.6, 0.7),
                       (np.array([0.1, 1.0, 0.2]), 0.4, 0.62)]:
        disp = u[None, :] + 0.12 * rng.standard_normal((6, 3))
        disp /= np.linalg.norm(disp, axis=1, keepdims=True)
        p = rng.uniform(0.5, 1.5, 6)
        p /= p.sum()
        for v, pv in zip(disp, p):
            weights.append(0.6 * ff * fin * pv)
            tensors.append(_rot_tensor([d_par, 0.0, 0.0], v))
            weights.append(0.6 * ff * (1 - fin) * pv)
            tensors.append(_rot_tensor([d_par, d_perp, d_perp], v))
    weights += [0.25, 0.15]
    tensors += [0.9e-3 * np.eye(3), 3.0e-3 * np.eye(3)]
    return np.array(weights), np.array(tensors)


def _mixture_signal(gtab, weights, tensors):
    w = weights / weights.sum()
    adc = np.einsum("cij,ni,nj->cn", tensors, gtab.bvecs, gtab.bvecs)
    return (w[:, None] * np.exp(-gtab.bvals[None, :] * adc)).sum(0)


def test_dki_params_exact_directional_kurtosis():
    """K(n) from analytic dki_params equals 3*Var/mean^2 of the mixture."""
    weights, tensors = _crossing_mixture()
    params = dki_params_from_tensor_distribution(weights, tensors)

    sphere = get_sphere(name="repulsion100")
    V = sphere.vertices
    K_dipy = apparent_kurtosis_coef(params[None], sphere, min_kurtosis=-1e9)[0]

    w = weights / weights.sum()
    adc = np.einsum("cij,ni,nj->cn", tensors, V, V)
    mean = (w[:, None] * adc).sum(0)
    var = (w[:, None] * adc ** 2).sum(0) - mean ** 2
    K_direct = 3.0 * var / mean ** 2
    assert_allclose(K_dipy, K_direct, atol=1e-10)


def test_moments_and_signal_reconstruction_from_odfs():
    """moments_from_odfs / synth_signal_from_odfs reproduce the exact mixture."""
    verts = default_sphere.vertices
    M = len(verts)
    d_par, d_perp = 2.3e-3, 0.55e-3
    gm_f, gm_d, csf_f, csf_d = 0.25, 0.9e-3, 0.15, 3.0e-3
    wm = 1 - gm_f - csf_f
    rng = np.random.default_rng(1)
    intra = np.zeros(M)
    extra = np.zeros(M)
    weights, tensors = [], []
    for u, ff, fin in [(np.array([1.0, 0.0, 0.0]), 0.6, 0.7),
                       (np.array([0.0, 1.0, 0.0]), 0.4, 0.6)]:
        cs = verts @ (u / np.linalg.norm(u))
        idx = np.argsort(np.abs(cs))[::-1][:5]
        p = rng.uniform(0.5, 1.5, 5)
        p /= p.sum()
        for j, pv in zip(idx, p):
            intra[j] += wm * ff * fin * pv
            extra[j] += wm * ff * (1 - fin) * pv
            weights += [wm * ff * fin * pv, wm * ff * (1 - fin) * pv]
            tensors += [_rot_tensor([d_par, 0, 0], verts[j]),
                        _rot_tensor([d_par, d_perp, d_perp], verts[j])]
    weights += [gm_f, csf_f]
    tensors += [gm_d * np.eye(3), csf_d * np.eye(3)]
    weights = np.array(weights)
    tensors = np.array(tensors)

    D_app, _ = moments_from_odfs(intra, extra, verts, d_par, d_perp,
                                 gm_f, gm_d, csf_f, csf_d)
    D_ref = np.einsum("c,cij->ij", weights / weights.sum(), tensors)
    assert_allclose(D_app[0], D_ref, atol=1e-12)

    gtab = _multishell_gtab([1000, 2000, 3000], n_dir=40, seed=2)
    S = synth_signal_from_odfs(intra, extra, verts, d_par, d_perp,
                               gm_f, gm_d, csf_f, csf_d,
                               gtab.bvals, gtab.bvecs)[0]
    S_ref = _mixture_signal(gtab, weights, tensors)
    assert_allclose(S, S_ref, atol=1e-12)


def test_analytic_dki_matches_fit_pure_wm():
    """Analytic cumulant DKI ~ a dipy DKI fit for a well-sampled pure-WM voxel
    (no free water -> the cumulant approximation is close at moderate b)."""
    d_par, d_perp = 2.0e-3, 0.4e-3
    rng = np.random.default_rng(3)
    weights, tensors = [], []
    u = np.array([1.0, 0.0, 0.0])
    disp = u[None, :] + 0.08 * rng.standard_normal((6, 3))
    disp /= np.linalg.norm(disp, axis=1, keepdims=True)
    for v in disp:
        weights += [0.7 / 6, 0.3 / 6]
        tensors += [_rot_tensor([d_par, 0, 0], v),
                    _rot_tensor([d_par, d_perp, d_perp], v)]
    weights = np.array(weights)
    tensors = np.array(tensors)
    params = dki_params_from_tensor_distribution(weights, tensors)

    gtab = _multishell_gtab([1000, 2000], n_dir=64, seed=4)
    S = _mixture_signal(gtab, weights, tensors)
    fit = dki.DiffusionKurtosisModel(gtab).fit(S)

    assert_allclose(dti.mean_diffusivity(params[:3]), fit.md, rtol=0.1)
    assert_allclose(dki.mean_kurtosis(params[None])[0], fit.mk(), rtol=0.25)


def test_mapmri_closed_forms_match_mapmrifit():
    """Closed-form RTOP/RTAP/RTPP/MSD/QIV match MapmriFit for a single
    well-conditioned anisotropic Gaussian."""
    verts = default_sphere.vertices
    M = len(verts)
    d_par, d_perp = 1.6e-3, 0.5e-3
    # single fibre along a vertex, all extra-axonal (zeppelin) -> fit-friendly
    j = int(np.argmax(verts @ np.array([1.0, 0.2, 0.1])))
    extra = np.zeros(M)
    extra[j] = 1.0
    intra = np.zeros(M)
    tensors = np.array([_rot_tensor([d_par, d_perp, d_perp], verts[j])])

    gtab = _multishell_gtab([1000, 2000, 3000, 4000, 5000], n_dir=90, seed=5)
    tau = gtab.big_delta - gtab.small_delta / 3.0
    S = _mixture_signal(gtab, np.array([1.0]), tensors)
    fit = mapmri.MapmriModel(gtab, radial_order=6,
                             laplacian_regularization=True,
                             laplacian_weighting="GCV").fit(S)
    idx = mapmri_closed_form_indices(intra, extra, verts, d_par, d_perp,
                                     0.0, 0.9e-3, 0.0, 3.0e-3, tau)
    assert_allclose(fit.rtop(), idx["rtop"][0], rtol=0.02)
    assert_allclose(fit.rtap(), idx["rtap"][0], rtol=0.02)
    assert_allclose(fit.rtpp(), idx["rtpp"][0], rtol=0.02)
    assert_allclose(fit.msd(), idx["msd"][0], rtol=0.02)
    assert_allclose(fit.qiv(), idx["qiv"][0], rtol=0.05)


def test_mapmri_floor_makes_stick_indices_finite():
    """RTOP/RTAP/QIV are finite for a stick-containing mixture with the floor."""
    verts = default_sphere.vertices
    M = len(verts)
    intra = np.zeros(M)
    extra = np.zeros(M)
    j = 10
    intra[j] = 0.6
    extra[j] = 0.2
    idx = mapmri_closed_form_indices(intra, extra, verts, 2.2e-3, 0.5e-3,
                                     0.1, 0.9e-3, 0.1, 3.0e-3, 0.04,
                                     d_perp_floor=0.12e-3)
    for k in ("rtop", "rtap", "rtpp", "msd", "qiv"):
        assert np.all(np.isfinite(idx[k]))
        assert idx[k][0] > 0


@pytest.mark.parametrize("method", ["canonical", "cumulant"])
def test_generate_single_shell_dki(method):
    """DKI (and MAP-MRI) can be generated from a SINGLE-SHELL scheme -- the
    case that failed with the legacy shell-based fit."""
    gtab = _multishell_gtab([1000], n_dir=48, seed=6)
    sims = generate_force_simulations(
        gtab, num_simulations=40, num_cpus=1, verbose=False,
        compute_dti=True, compute_dki=True, compute_mapmri=True,
        metric_method=method,
    )
    for k in ("fa", "md", "ak", "rk", "mk", "mkt", "kfa", "rtop", "msd", "qiv"):
        assert np.all(np.isfinite(sims[k])), k
    assert np.all(sims["md"] > 0)


def test_rotational_invariance():
    """Rotation-invariant scalars (MD, MK, KFA, FA) are unchanged when the
    whole mixture is rotated -- confirms the xyz-frame kt + evecs assembly."""
    weights, tensors = _crossing_mixture(seed=11)
    # random rotation
    rng = np.random.default_rng(12)
    A = rng.standard_normal((3, 3))
    R, _ = np.linalg.qr(A)
    tensors_rot = np.einsum("ij,cjk,lk->cil", R, tensors, R)

    p0 = dki_params_from_tensor_distribution(weights, tensors)
    p1 = dki_params_from_tensor_distribution(weights, tensors_rot)
    assert_allclose(dti.mean_diffusivity(p0[:3]),
                    dti.mean_diffusivity(p1[:3]), rtol=1e-6)
    assert_allclose(dti.fractional_anisotropy(p0[:3]),
                    dti.fractional_anisotropy(p1[:3]), rtol=1e-6)
    assert_allclose(dki.mean_kurtosis(p0[None])[0],
                    dki.mean_kurtosis(p1[None])[0], rtol=1e-4)
    assert_allclose(dki.kurtosis_fractional_anisotropy(p0[None])[0],
                    dki.kurtosis_fractional_anisotropy(p1[None])[0], rtol=1e-4)


def test_covariance_matches_qti_equal_weight():
    """The Cartesian covariance used for the kurtosis tensor matches dipy's
    (unmodified) qti.dtd_covariance for equal weights (Voigt <-> Cartesian)."""
    from dipy.reconst.qti import dtd_covariance, from_6x6_to_21x1

    rng = np.random.default_rng(13)
    _, tensors = _crossing_mixture(seed=13)
    tensors = tensors[:8]
    n = len(tensors)
    w = np.full(n, 1.0 / n)
    D_app = np.einsum("c,cij->ij", w, tensors)
    C = (np.einsum("c,cij,ckl->ijkl", w, tensors, tensors)
         - np.einsum("ij,kl->ijkl", D_app, D_app))
    # qti covariance (6x6 Voigt) with equal weights
    C_qti = dtd_covariance(tensors)  # (6,6)
    v = np.sqrt(2.0)
    # map Cartesian rank-4 -> Voigt 6x6 to compare diagonal (xx,yy,zz) block
    cart = {0: (0, 0), 1: (1, 1), 2: (2, 2)}
    for a in range(3):
        for b in range(3):
            i, j = cart[a]
            k, l = cart[b]
            assert_allclose(C[i, j, k, l], C_qti[a, b], atol=1e-12)


def test_ng_pa_archetypes():
    """Closed-form NG/PA behave correctly on microstructural archetypes."""
    I = np.eye(3)
    # single anisotropic Gaussian -> NG=0, PA>0
    Z = _rot_tensor([1.6e-3, 0.5e-3, 0.5e-3], [1, 0, 0])
    r = gaussian_mixture_ng_pa([1.0], [Z])
    assert r["ng"] < 1e-6 and r["ngperp"] < 1e-6
    assert r["pa"] > 0.05
    # free water (isotropic Gaussian) -> NG=0, PA=0
    r = gaussian_mixture_ng_pa([1.0], [3.0e-3 * I])
    assert r["ng"] < 1e-6 and r["pa"] < 1e-6
    # restriction (stick + zeppelin) -> NG>0 concentrated in ngperp, ngpar~0
    St = _rot_tensor([1.8e-3, 0, 0], [1, 0, 0])
    r = gaussian_mixture_ng_pa([0.6, 0.4], [St, Z])
    assert r["ng"] > 0.05
    assert r["ngperp"] > 5 * (r["ngpar"] + 1e-6)
    # adding free water raises NG monotonically for a zeppelin
    ngs = [gaussian_mixture_ng_pa([1 - f, f], [Z, 3.0e-3 * I])["ng"]
           for f in (0.0, 0.15, 0.3)]
    assert ngs[0] < ngs[1] < ngs[2]


def test_generate_compute_ng():
    """compute_ng stores finite ng/ngpar/ngperp/pa in [0, 1)."""
    gtab = _multishell_gtab([1000], n_dir=40, seed=30)
    sims = generate_force_simulations(
        gtab, num_simulations=25, num_cpus=1, verbose=False,
        compute_dti=False, compute_dki=False, compute_ng=True,
    )
    for k in ("ng", "ngpar", "ngperp", "pa"):
        assert k in sims
        assert np.all(np.isfinite(sims[k]))
        assert np.all((sims[k] >= 0) & (sims[k] <= 1.0001)), k


@pytest.mark.parametrize("model,keys", [
    ("mapmri", ("rtop", "rtap", "rtpp", "msd", "qiv", "ng", "ngpar", "ngperp")),
    ("shore", ("rtop", "msd")),
])
def test_generate_mapmri_canonical_fit(model, keys):
    """MAP-MRI/SHORE indices via a fit to a synthesized canonical scheme."""
    gtab = _multishell_gtab([1000, 2000], n_dir=40, seed=20)
    sims = generate_force_simulations(
        gtab, num_simulations=12, num_cpus=1, verbose=False,
        compute_dti=False, compute_dki=False, compute_mapmri=True,
        mapmri_method="canonical", mapmri_fit_model=model,
        mapmri_canonical_bvals=(1000.0, 2000.0, 3000.0),
    )
    for k in keys:
        assert k in sims and np.all(np.isfinite(sims[k])), k
    # closed-form-only keys absent for shore
    if model == "shore":
        assert "qiv" not in sims


def test_qti_invariants_archetypes():
    """QTI/DIVIDE invariants from the covariance behave correctly."""
    from dipy.reconst import dti

    def _mix(weights, tensors):
        w = np.asarray(weights, float) / np.sum(weights)
        T = np.asarray(tensors, float)
        D = np.einsum("c,cij->ij", w, T)
        C = np.einsum("c,cij,ckl->ijkl", w, T, T) - np.einsum("ij,kl->ijkl", D, D)
        return D[None], C[None]

    Z = _rot_tensor([1.6e-3, 0.4e-3, 0.4e-3], [1, 0, 0])
    St = _rot_tensor([1.8e-3, 0.12e-3, 0.12e-3], [1, 0, 0])
    Xc = _rot_tensor([1.8e-3, 0.12e-3, 0.12e-3], [0, 1, 0])
    # single anisotropic Gaussian: micro_fa == FA, coherence == 1, no kurtosis
    q = qti_indices_from_moments(*_mix([1], [Z]))
    assert np.isclose(q["micro_fa"][0],
                      dti.fractional_anisotropy(np.linalg.eigvalsh(Z)), atol=1e-6)
    assert np.isclose(q["coherence"][0], 1.0, atol=1e-6)
    assert q["k_bulk"][0] < 1e-9 and q["k_shear"][0] < 1e-9
    # free water: finite (not NaN), micro_fa ~ 0
    q = qti_indices_from_moments(*_mix([1], [3.0e-3 * np.eye(3)]))
    assert np.all(np.isfinite([q[k][0] for k in q]))
    assert q["micro_fa"][0] < 1e-6
    # fibre + free water: isotropic (bulk) kurtosis dominates the split
    q = qti_indices_from_moments(*_mix([0.6, 0.4], [St, 3.0e-3 * np.eye(3)]))
    assert q["k_bulk"][0] > q["k_shear"][0]
    # crossing: micro_fa stays high while coherence drops below 1
    q = qti_indices_from_moments(*_mix([0.5, 0.5], [St, Xc]))
    assert q["micro_fa"][0] > 0.8 and q["coherence"][0] < 0.7


def test_generate_compute_qti():
    """compute_qti stores finite micro_fa/coherence/k_bulk/k_shear."""
    gtab = _multishell_gtab([1000, 2000], n_dir=30, seed=11)
    sims = generate_force_simulations(
        gtab, num_simulations=25, num_cpus=1, verbose=False,
        compute_dti=False, compute_dki=False, compute_qti=True,
    )
    for k in ("micro_fa", "coherence", "k_bulk", "k_shear"):
        assert k in sims and np.all(np.isfinite(sims[k])), k
    assert np.all((sims["micro_fa"] >= 0) & (sims["micro_fa"] <= 1.001))
    assert np.all((sims["coherence"] >= 0) & (sims["coherence"] <= 1.001))


def test_generate_signal_method_shelled():
    """Legacy metric_method='signal' still works on a well-shelled scheme
    (guards the extracted _signal_fit_scalars helper and its imports)."""
    gtab = _multishell_gtab([1000, 2000], n_dir=30, seed=8)
    sims = generate_force_simulations(
        gtab, num_simulations=30, num_cpus=1, verbose=False,
        compute_dti=True, compute_dki=True, metric_method="signal",
    )
    for k in ("fa", "md", "rd", "ak", "rk", "mk", "kfa"):
        assert np.all(np.isfinite(sims[k])), k


@pytest.mark.parametrize("scheme", ["cartesian", "random"])
def test_generate_nonshelled_no_error(scheme):
    """Cartesian and random q-space schemes complete without error."""
    rng = np.random.default_rng(7)
    if scheme == "cartesian":
        g = np.arange(-2, 3)
        pts = np.array([[x, y, z] for x in g for y in g for z in g
                        if (x, y, z) != (0, 0, 0)], float)
        bvals = 250.0 * (pts ** 2).sum(1)
        bvecs = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    else:
        bvals = rng.uniform(0, 3000, 150)
        bvecs = rng.standard_normal((150, 3))
        bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)
    b = np.concatenate([[0.0], bvals])
    v = np.vstack([[0, 0, 0], bvecs])
    gtab = gradient_table(b, bvecs=v, big_delta=0.043, small_delta=0.0106)
    sims = generate_force_simulations(
        gtab, num_simulations=30, num_cpus=1, verbose=False,
        compute_dti=True, compute_dki=True, metric_method="canonical",
    )
    assert np.all(np.isfinite(sims["mk"]))
    assert np.all(np.isfinite(sims["fa"]))
