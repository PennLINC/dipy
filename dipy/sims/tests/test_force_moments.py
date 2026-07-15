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
    RSI_KEYS,
    GQI_KEYS,
    dki_params_from_moments,
    dki_params_from_tensor_distribution,
    gaussian_mixture_ng_pa,
    gqi_indices_from_odfs,
    mapmri_closed_form_indices,
    moments_from_odfs,
    qti_indices_from_moments,
    rsi_indices_from_signals,
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


def _abcd_reference_rsi(signals, gtab):
    """Independent re-implementation of the ABCD ``RSIproc`` estimator using
    ``scipy.special.sph_harm`` (the reference's own basis) on the same vendored
    icosphere, so the test does not depend on the production real-SH convention.
    Returns the same dict of keys as ``rsi_indices_from_signals``."""
    from dipy.sims._force_moments import _rsi_icosahedron, RSI_COMPARTMENTS
    try:                                    # scipy >= 1.15 renamed sph_harm
        from scipy.special import sph_harm_y

        def _Y(m, l, az, pol):
            return sph_harm_y(l, m, pol, az)
    except ImportError:
        from scipy.special import sph_harm

        def _Y(m, l, az, pol):
            return sph_harm(m, l, az, pol)

    def yl(l, ml, x):                       # RSIproc YLeval
        pol = np.arccos(x[2]); az = np.arctan2(x[1], x[0])
        az = az + (az < 0) * 2 * np.pi
        y = _Y(abs(ml), l, az, pol)
        if ml < 0:
            return np.sqrt(2) * y.real
        if ml == 0:
            return y.real
        return np.sqrt(2) * y.imag

    X = _rsi_icosahedron(3); M = X.shape[0]
    orders = [c[3] for c in RSI_COMPARTMENTS]
    max_order = max(orders)
    nshmax = (max_order + 2) * (max_order + 1) // 2
    YL = np.zeros((M, nshmax))
    l_of_p = np.zeros(nshmax, int)              # RSIproc even-l contiguous packing
    for l in range(0, max_order + 2, 2):
        for ml in range(-l, l + 1):
            p = (l * l + l) // 2 + ml
            YL[:, p] = [yl(l, ml, X[m]) for m in range(M)]
            l_of_p[p] = l
    bvals = np.asarray(gtab.bvals, float); bvecs = np.asarray(gtab.bvecs, float)
    Ls, cids, blocks = [], [], []
    for ci, (_, DL, DT, sho) in enumerate(RSI_COMPARTMENTS):
        nsh = (sho + 2) * (sho + 1) // 2
        R = np.exp(-bvals[:, None] * (DT + (DL - DT) * (bvecs @ X.T) ** 2))
        blocks.append(R @ np.linalg.pinv(YL[:, :nsh]).T)
        Ls.append(l_of_p[:nsh]); cids.append(np.full(nsh, ci))
    A = np.hstack(blocks) * YL[0, 0]
    L = np.concatenate(Ls); cid = np.concatenate(cids)
    AtA = A.T @ A
    W = np.linalg.solve(AtA + 0.1 * np.mean(np.diag(AtA)) * np.eye(A.shape[1]), A.T)
    b0 = bvals <= 50
    S0 = signals[:, b0].mean(1)
    beta = (signals / np.maximum(S0, 1e-9)[:, None]) @ W.T
    out = {}
    names = ["r", "h", "f"]
    for ci, nm in enumerate(names):
        b = beta[:, cid == ci]; li = L[cid == ci]
        n0 = np.clip(b[:, li == 0][:, 0], -1, 2)
        nd = np.linalg.norm(b[:, li > 0], axis=1) if np.any(li > 0) else np.zeros(len(b))
        nt = np.linalg.norm(b, axis=1)
        out[nm] = (n0, nd, nt)
    r, h, f = out["r"], out["h"], out["f"]
    return {"rnt": r[2], "rn0": r[0], "rnd": r[1],
            "hnt": h[2], "hn0": h[0], "hnd": h[1], "fnt": f[2]}


def test_rsi_indices_match_abcd_and_generate():
    """RSI reproduces the ABCD RSIproc convention: the production estimator
    matches an independent scipy-``sph_harm`` re-implementation of RSIproc, the
    stored ``compute_rsi`` columns equal the direct projection, and per
    compartment ``NT**2 = N0**2 + ND**2``."""
    gtab = _multishell_gtab([500, 1000, 2000, 3000], n_dir=60, seed=3)
    sims = generate_force_simulations(
        gtab, num_simulations=40, num_cpus=1, verbose=False,
        compute_dti=False, compute_dki=False, compute_rsi=True,
    )
    for k in RSI_KEYS:
        assert k in sims and np.all(np.isfinite(sims[k])), k
    # ranges: N0 clipped to [-1, 2]; NT/ND are non-negative norms
    for k in ("rn0", "hn0"):
        assert np.all((sims[k] >= -1.0001) & (sims[k] <= 2.0001)), k
    for k in ("rnt", "rnd", "hnt", "hnd", "fnt"):
        assert np.all(sims[k] >= -1e-9), k
    # NT**2 = N0**2 + ND**2 per directional compartment
    assert np.allclose(sims["rnt"] ** 2, sims["rn0"] ** 2 + sims["rnd"] ** 2, atol=1e-5)
    assert np.allclose(sims["hnt"] ** 2, sims["hn0"] ** 2 + sims["hnd"] ** 2, atol=1e-5)
    # stored (exact) == direct projection of the same signals
    r = rsi_indices_from_signals(sims["signals"], gtab)
    for k in RSI_KEYS:
        assert np.allclose(sims[k], r[k], atol=1e-5), k
    # production real-SH estimator == independent scipy-sph_harm ABCD reference
    ref = _abcd_reference_rsi(np.asarray(sims["signals"], float), gtab)
    for k in RSI_KEYS:
        assert_allclose(r[k], ref[k], atol=2e-4, rtol=2e-3,
                        err_msg=f"{k} differs from ABCD RSIproc reference")


def test_rish_features_rotation_invariance():
    """RISH per-shell SH power is far more rotation-stable than the raw signal:
    rotating the microstructure leaves RISH ~unchanged while the raw signal moves
    substantially. Also checks generate stores the columns."""
    from dipy.sims._force_moments import (rish_features_from_signals,
                                           synth_signal_from_odfs)
    sph = get_sphere(name="repulsion724")
    verts = np.asarray(sph.vertices, float)
    gtab = _multishell_gtab([1000, 2000, 3000], n_dir=60, seed=2)
    bvals, bvecs = np.asarray(gtab.bvals), np.asarray(gtab.bvecs)

    def watson(u, odi, frac):
        k = 1.0 / np.tan(np.pi / 2 * odi)
        w = np.exp(k * (verts @ u) ** 2)
        return frac * w / w.sum()

    def rot(seed):
        q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((3, 3)))
        return q * np.sign(np.linalg.det(q))

    args = (2.2e-3, 0.6e-3, 0.18, 1.0e-3, 0.12, 3.0e-3)
    u1 = np.array([0, 0, 1.0]); u2 = np.array([np.sin(1.0), 0, np.cos(1.0)])

    def build(R):
        io = (watson(R @ u1, 0.1, 0.3) + watson(R @ u2, 0.1, 0.2))[None]
        eo = (watson(R @ u1, 0.1, 0.12) + watson(R @ u2, 0.1, 0.08))[None]
        return synth_signal_from_odfs(io, eo, verts, *args, bvals, bvecs)

    S0 = build(np.eye(3))
    base = rish_features_from_signals(S0, gtab)
    feats, cos = {k: [float(base[k])] for k in base}, []
    for s in range(1, 12):
        Sr = build(rot(s))
        r = rish_features_from_signals(Sr, gtab)
        for k in feats:
            feats[k].append(float(r[k]))
        cos.append(float(S0 @ Sr.T / (np.linalg.norm(S0) * np.linalg.norm(Sr))))
    # low-order RISH power (l<=2) barely moves; the raw signal moves a lot
    lo_spread = max(np.std(feats[k]) / (abs(np.mean(feats[k])) + 1e-9)
                    for k in feats if k.endswith(("l0", "l2")))
    raw_move = 1.0 - np.mean(cos)
    assert lo_spread < 0.06                     # RISH l<=2 stable under rotation
    assert raw_move > lo_spread                 # raw signal moves more than RISH

    sims = generate_force_simulations(gtab, num_simulations=20, num_cpus=1,
                                      verbose=False, compute_dti=True,
                                      compute_rish=True)
    rk = [k for k in sims if k.startswith("rish_")]
    assert len(rk) >= 6 and all(np.all(np.isfinite(sims[k])) for k in rk)


def test_num_fiber_probs():
    """num_fiber_probs re-weights the WM fibre-count mix (default unchanged)."""
    gtab = _multishell_gtab([1000, 2000], n_dir=24, seed=1)
    common = dict(num_simulations=1500, num_cpus=1, verbose=False, compute_dti=True)
    # default is 3-fibre-heavy
    d = generate_force_simulations(gtab, **common)
    nf = np.asarray(d["num_fibers"])
    assert (nf == 3).mean() > (nf == 2).mean()          # default favours 3 fibres
    # rebalanced toward 2-fibre crossings
    r = generate_force_simulations(gtab, num_fiber_probs=(0.15, 0.6, 0.25), **common)
    nf2 = np.asarray(r["num_fibers"])
    assert (nf2 == 2).mean() > 0.45 and (nf2 == 2).mean() > (nf == 2).mean()
    # pure two-fibre
    p2 = generate_force_simulations(gtab, num_fiber_probs=(0, 1, 0), **common)
    assert np.all(np.asarray(p2["num_fibers"]) == 2)


def test_soma_compartment():
    """The optional soma/dot compartment: off by default (no regression), and
    when on it is an exact 4th isotropic Gaussian -- weights still sum to 1, the
    mixture dilutes MD, and RSI's restricted-isotropic term (rn0) picks it up."""
    gtab = _multishell_gtab([1000, 2000, 3000], n_dir=50, seed=11)
    kw = dict(num_simulations=250, num_cpus=1, verbose=False,
              compute_dti=True, compute_dki=True, metric_method="cumulant",
              compute_mapmri=True, compute_qti=True, compute_rsi=True)

    np.random.seed(3)
    off = generate_force_simulations(gtab, **kw)
    np.random.seed(3)
    on = generate_force_simulations(gtab, include_soma=True, **kw)

    # off by default -> no soma columns, behaviour unchanged
    assert "soma_fraction" not in off and "soma_d" not in off
    assert "soma_fraction" in on and "soma_d" in on

    fs = np.asarray(on["soma_fraction"])
    # the mixture is still a valid partition of unity
    total = (np.asarray(on["wm_fraction"]) + np.asarray(on["gm_fraction"])
             + np.asarray(on["csf_fraction"]) + fs)
    assert_allclose(total, 1.0, atol=1e-5)
    # every analytic scalar stays finite with the extra compartment
    for k in ("fa", "md", "rd", "mk", "rtop", "rtap", "msd", "qiv",
              "micro_fa", "rn0", "rnt", "fnt"):
        assert np.all(np.isfinite(on[k])), k

    # a low-diffusivity isotropic compartment must pull MD down ...
    assert np.corrcoef(fs, np.asarray(on["md"]))[0, 1] < -0.15
    # ... and show up in RSI's restricted-ISOTROPIC term, which is the whole
    # point of adding it (it otherwise has no generative counterpart).
    assert np.corrcoef(fs, np.asarray(on["rn0"]))[0, 1] > 0.25


def test_soma_moments_match_explicit_mixture():
    """moments_from_odfs with a soma compartment equals an explicit 4-compartment
    tensor distribution (the soma term is exact, not an approximation)."""
    rng = np.random.default_rng(0)
    verts = np.asarray(get_sphere(name="repulsion724").vertices, float)
    n_v = len(verts)
    io = np.zeros((1, n_v)); eo = np.zeros((1, n_v))
    idx = rng.choice(n_v, 4, replace=False)
    io[0, idx] = [0.10, 0.06, 0.04, 0.02]      # intra weights
    eo[0, idx] = [0.05, 0.03, 0.02, 0.01]      # extra weights
    d_par, d_perp = 2.2e-3, 0.6e-3
    gm_f, gm_d, csf_f, csf_d = 0.15, 1.0e-3, 0.12, 3.0e-3
    soma_f, soma_d = 1.0 - (io.sum() + eo.sum() + gm_f + csf_f), 0.3e-3
    assert soma_f > 0

    D_app, C = moments_from_odfs(io, eo, verts, d_par, d_perp,
                                 gm_f, gm_d, csf_f, csf_d,
                                 soma_frac=soma_f, soma_d=soma_d)
    # explicit reference mixture
    w, T = [], []
    for i in idx:
        v = verts[i]; vv = np.outer(v, v)
        w.append(io[0, i]); T.append(d_par * vv)
        w.append(eo[0, i]); T.append(d_perp * np.eye(3) + (d_par - d_perp) * vv)
    for f, d in ((gm_f, gm_d), (csf_f, csf_d), (soma_f, soma_d)):
        w.append(f); T.append(d * np.eye(3))
    w = np.asarray(w); T = np.asarray(T)
    assert_allclose(w.sum(), 1.0, atol=1e-8)
    D_ref = np.einsum("c,cij->ij", w, T)
    DD_ref = np.einsum("c,cij,ckl->ijkl", w, T, T)
    C_ref = DD_ref - np.einsum("ij,kl->ijkl", D_ref, D_ref)
    assert_allclose(D_app[0], D_ref, atol=1e-12)
    assert_allclose(C[0], C_ref, atol=1e-14)


def test_gqi_indices_archetypes_and_generate():
    """GQI GFA/QA: isotropic ODF -> 0, sharp single-peak -> high GFA, and QA
    grows with peak sharpness; compute_gqi stores finite in-range measures."""
    n_vert = 200
    rng = np.random.default_rng(0)
    verts = rng.standard_normal((n_vert, 3))
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    # isotropic ODF (all WM) -> GFA/QA ~ 0
    iso = np.ones((1, n_vert))
    g_iso = gqi_indices_from_odfs(iso, np.array([1.0]))
    assert g_iso["gfa"][0] < 1e-6 and g_iso["qa"][0] < 1e-6
    # increasingly sharp single peak (all WM) -> GFA and QA increase monotonically
    axis = np.array([0.0, 0.0, 1.0])
    cos = verts @ axis
    odfs = np.stack([np.exp(kappa * cos**2) for kappa in (1.0, 5.0, 20.0)])
    g = gqi_indices_from_odfs(odfs, np.ones(3))
    assert np.all(np.diff(g["gfa"]) > 0) and np.all(np.diff(g["qa"]) > 0)
    # free-water dilution: same sharp ODF with low WM fraction -> lower GFA
    sharp = odfs[-1:][None, 0]
    g_hi = gqi_indices_from_odfs(sharp, np.array([1.0]))["gfa"][0]
    g_lo = gqi_indices_from_odfs(sharp, np.array([0.2]))["gfa"][0]
    assert g_lo < g_hi

    gtab = _multishell_gtab([1000, 2000, 3000], n_dir=40, seed=5)
    sims = generate_force_simulations(
        gtab, num_simulations=30, num_cpus=1, verbose=False,
        compute_dti=True, compute_dki=False, compute_gqi=True,
    )
    for k in GQI_KEYS:
        assert k in sims and np.all(np.isfinite(sims[k])), k
    assert np.all((sims["gfa"] >= -1e-9) & (sims["gfa"] <= 1.0001))
    assert np.all(sims["qa"] >= -1e-9)


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
