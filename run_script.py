# Julia signal/thread settings must be set before juliacall is initialized.
# Use os.environ[] (not setdefault) to override the THREADS=8 that
# jutul_darcy.py sets at module level, because fork() inside a multi-threaded
# Julia process causes a segfault via p_tqdm.
import os
os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
os.environ["PYTHON_JULIACALL_THREADS"] = "1"
os.environ["PYTHON_JULIACALL_OPTLEVEL"] = "3"

import copy
import numpy as np
from scipy.sparse import eye as speye
from scipy.sparse.linalg import LinearOperator, cg
from subsurface.multphaseflow.jutul_darcy import JutulDarcy
from input_output import read_config
import jutuldarcy as jd
import pandas as pd
from sksparse.cholmod import cho_factor

# ============================================================
# Prior definition (directly on log_permx)
# ============================================================
log_mean = np.log(500)  # prior mean in log-space
log_std = 0.5           # prior std in log-space (approx ±1 order of magnitude at 2σ)

SEED = 0  # for reproducibility
LAPLACE_EIG_TOL = 1e-5

Q_prior, mask, geom = jd.matern_precision_from_data_file(
    '/home/AD.NORCERESEARCH.NO/krfo/CodeProjects/makeSENSE/5SPOT/TRUE_MODEL/TRUE_MODEL.DATA',
    layer=1,
    target_variance=log_std**2,
    target_range=2500.0,
    target_anisotropy=2.0,
    target_rotation=45.0,
    return_mask=True,
    return_geometry=True
)

n_x = Q_prior.shape[0]
mu_prior = np.full(n_x, log_mean)
mask_arr = np.asarray(mask)
flat_active_mask = mask_arr.ravel(order='F') == 1

# Precompute sparse Cholesky of Q_prior (reused throughout)
L_prior = cho_factor(Q_prior, lower=True)


def active_vector_from_sample(sample, *, name="sample"):
    """
    Convert a prior sample returned by jd.sample_from_precision to an active-cell
    vector in the same Fortran-order convention used by Q_prior and JutulDarcy.
    """
    if np.ma.isMaskedArray(sample):
        data = np.asarray(sample.filled(np.nan), dtype=float)
    else:
        data = np.asarray(sample, dtype=float)

    if data.shape == mask_arr.shape:
        vec = data.ravel(order='F')[flat_active_mask]
    else:
        flat = data.ravel(order='F')
        if flat.size == n_x:
            vec = flat
        elif flat.size == mask_arr.size:
            vec = flat[flat_active_mask]
        else:
            raise ValueError(
                f"{name} has shape {data.shape} / size {flat.size}, "
                f"expected active vector length {n_x} or grid shape {mask_arr.shape}"
            )

    vec = np.asarray(vec, dtype=float)
    if vec.shape != (n_x,):
        raise ValueError(f"{name} active vector shape {vec.shape} != {(n_x,)}")
    if not np.all(np.isfinite(vec)):
        raise ValueError(f"{name} contains non-finite values in active cells")
    return vec


# ============================================================
# Simulator setup
# ============================================================
_, kwsim, _ = read_config.read('config.yaml')
sim = JutulDarcy(kwsim)
kwsim_pred = copy.deepcopy(kwsim)
kwsim_pred.pop('adjoints', None)
kwsim_pred['parallel'] = 1
sim_pred = JutulDarcy(kwsim_pred)

# ============================================================
# Load observations and build data covariance
# ============================================================
data_df = pd.read_pickle('DATA/true_data.pkl')
var_df = pd.read_pickle('DATA/true_data_var.pkl')

# Flatten data and variance into vectors aligned with adjoint columns
# data_df columns are like 'WOPR:P1', 'WGPR:P2', etc.
# adjoint columns are MultiIndex (objective, parameter) where objective = 'WOPR:P1'
d_obs = data_df.values.flatten(order='F')  # stack columns: all dates for col0, then col1, ...
n_d = len(d_obs)

# Build variance vector (same ordering)
# var_df contains ['abs', variance_value] per cell
cd_diag = np.zeros(n_d)
idx = 0
for col in data_df.columns:
    for i, row in enumerate(data_df.index):
        entry = var_df.loc[row, col]
        if isinstance(entry, list):
            cd_diag[idx] = entry[1]
        else:
            cd_diag[idx] = float(entry)
        idx += 1

cd_inv = 1.0 / cd_diag
cd_inv_sqrt = np.sqrt(cd_inv)


def assemble_prediction_and_jacobian(pred_df, adj_df):
    """
    Assemble prediction vector and Jacobian matrix from simulator DataFrames.

    Returns:
        g: prediction vector (n_d,) aligned with d_obs
        J: Jacobian matrix (n_d, n_x)
    """
    g = np.zeros(n_d)
    J = np.zeros((n_d, n_x))

    idx = 0
    for col in data_df.columns:
        n_t = len(data_df.index)
        # Prediction
        g[idx:idx + n_t] = pred_df[col].values

        # Jacobian: adj_df has MultiIndex columns (objective, parameter)
        # objective matches data column name, parameter is 'log_permx'
        # Each cell contains a gradient vector of length n_x
        if (col, 'log_permx') in adj_df.columns:
            J[idx:idx + n_t, :] = np.vstack(adj_df[(col, 'log_permx')].values)
        idx += n_t

    return g, J


def assemble_prediction(pred_df):
    """Assemble simulator predictions into the same F-order vector as d_obs."""
    g = np.zeros(n_d)
    idx = 0
    for col in data_df.columns:
        n_t_col = len(data_df.index)
        g[idx:idx + n_t_col] = pred_df[col].values
        idx += n_t_col
    return g


def compute_predictions(x):
    """Run a forward-only simulation and return predictions aligned with d_obs."""
    pred_df = sim_pred({'log_permx': x})
    return assemble_prediction(pred_df)


def objective(x, g=None):
    """Compute MAP objective value."""
    dx = x - mu_prior
    prior_term = 0.5 * dx @ (Q_prior @ dx)
    if g is None:
        return prior_term
    r = g - d_obs
    likelihood_term = 0.5 * np.sum(cd_inv * r**2)
    return prior_term + likelihood_term


# ============================================================
# Phase B: MAP optimization (Gauss-Newton with LM damping)
# ============================================================
max_iter = 10
lam = 1000.0   # initial LM damping
lam_min = 1e-3
lam_factor = 10.0
tol_grad = 1e-4
tol_step = 1e-6

rng = np.random.default_rng(SEED)
# Initial guess: prior sample
prior_sample = jd.sample_from_precision(Q_prior, mask, n_samples=1, rng=rng)
x_k = mu_prior + active_vector_from_sample(prior_sample, name="initial prior sample")

print(f"{'Iter':>4} {'Objective':>12} {'|grad|':>12} {'lambda':>10} {'step_norm':>10}")
print("-" * 60)

# Forward simulation + adjoint
pred_k, adj_df = sim({'log_permx': x_k})
g_k, J_k = assemble_prediction_and_jacobian(pred_k, adj_df)

obj_prev = np.inf
for k in range(max_iter):
    # Objective
    obj_k = objective(x_k, g_k)

    # Gradient: Q_prior @ (x - mu) + J^T @ C_d^{-1} @ (g - d)
    r_k = g_k - d_obs
    grad_k = Q_prior @ (x_k - mu_prior) + J_k.T @ (cd_inv * r_k)
    grad_norm = np.linalg.norm(grad_k)

    print(f"{k:4d} {obj_k:12.4e} {grad_norm:12.4e} {lam:10.2e} {'---':>10}" if k == 0
          else f"{k:4d} {obj_k:12.4e} {grad_norm:12.4e} {lam:10.2e} {step_norm:10.4e}")

    # Check convergence
    if grad_norm < tol_grad:
        print("Converged: gradient norm below tolerance.")
        break

    # Whitened Jacobian
    J_tilde = cd_inv_sqrt[:, None] * J_k  # (n_d, n_x), scale each row

    # GN step with LM damping: solve (Q_prior + lam*I + J_tilde^T J_tilde) dx = -grad
    Q_damped = (Q_prior + lam * speye(n_x, format='csc')).tocsc()
    L_damped = cho_factor(Q_damped, lower=True)

    # Woodbury with damped prior
    z1 = L_damped.solve(-grad_k)
    Q_inv_Jt = L_damped.solve(J_tilde.T)
    M = np.eye(n_d) + J_tilde @ Q_inv_Jt
    z2 = J_tilde @ z1
    z3 = np.linalg.solve(M, z2)
    delta_x = z1 - Q_inv_Jt @ z3

    step_norm = np.linalg.norm(delta_x)

    # Evaluate candidate
    x_cand = x_k + delta_x
    pred_cand, adj_cand = sim({'log_permx': x_cand})
    g_cand, J_cand = assemble_prediction_and_jacobian(pred_cand, adj_cand)
    obj_cand = objective(x_cand, g_cand)

    # Accept/reject with LM strategy
    if obj_cand < obj_k:
        x_k = x_cand
        obj_prev = obj_k
        pred_k, adj_df = pred_cand, adj_cand
        g_k, J_k = g_cand, J_cand
        # Decrease damping
        lam = max(lam / lam_factor, lam_min)
    else:
        # Increase damping and retry
        lam *= lam_factor
        print(f"     Step rejected, increasing lambda to {lam:.2e}")

    if step_norm < tol_step:
        print("Converged: step norm below tolerance.")
        break

x_MAP = x_k
print(f"\nMAP found. Final objective: {objective(x_MAP, g_k):.6e}")

# ============================================================
# Phase C: GN Hessian at MAP and posterior operations
# ============================================================
# Final forward + adjoint at MAP (may already be computed if last step was accepted)
pred_map, adj_map = sim({'log_permx': x_MAP})
g_map, J_map = assemble_prediction_and_jacobian(pred_map, adj_map)
J_tilde_map = cd_inv_sqrt[:, None] * J_map  # whitened Jacobian at MAP

# Posterior precision: Q_post = Q_prior + J_tilde_map^T @ J_tilde_map
# Stored implicitly as (Q_prior, J_tilde_map)

# Precompute for posterior operations
Q_inv_Jt_map = L_prior.solve(J_tilde_map.T)  # (n_x, n_d)
M_post = np.eye(n_d) + J_tilde_map @ Q_inv_Jt_map  # (n_d, n_d)


# ============================================================
# Prior-whitened matrix-free posterior sampling
# ============================================================
# Posterior:  x | d ~ N(x_MAP, H_post^{-1})
# with        H_post = Q_prior + J_tilde^T J_tilde
#
# To draw s ~ N(0, H_post^{-1}) matrix-free:
#   x_p  ~ N(0, Q_prior^{-1})        (via jd.sample_from_precision)
#   eps  ~ N(0, I_{n_d})
#   rhs  = Q_prior @ x_p + J_tilde^T @ eps   -> Cov(rhs) = H_post
#   solve  H_post @ s = rhs   via preconditioned CG
# Preconditioning by Q_prior^{-1} is the prior-whitening: the preconditioned
# operator is  I + B^T B  with  B = J_tilde @ L_prior^{-T}, which is well
# conditioned (eigenvalues >= 1 with only n_d eigenvalues > 1).

def _H_post_matvec(v):
    return Q_prior @ v + J_tilde_map.T @ (J_tilde_map @ v)


def _M_prec_matvec(v):
    return L_prior.solve(v)


H_post_op = LinearOperator((n_x, n_x), matvec=_H_post_matvec, dtype=float)
M_prec_op = LinearOperator((n_x, n_x), matvec=_M_prec_matvec, dtype=float)


def sample_posterior(n_samples=1, rng=None, cg_tol=1e-6, cg_maxiter=200, verbose=False):
    """
    Matrix-free, prior-whitened sampling from the Laplace posterior
    N(x_MAP, (Q_prior + J_tilde^T J_tilde)^{-1}).
    """
    if rng is None:
        rng = np.random.default_rng()

    raw_prior_samples = jd.sample_from_precision(
        Q_prior, mask, n_samples=n_samples, rng=rng,
    )
    if not isinstance(raw_prior_samples, list):
        raw_prior_samples = [raw_prior_samples]
    if len(raw_prior_samples) != n_samples:
        raise ValueError(
            f"jd.sample_from_precision returned {len(raw_prior_samples)} samples, "
            f"expected {n_samples}"
        )

    # Prior samples ~ N(0, Q_prior^{-1}), active cells in Q/JutulDarcy order.
    x_p = np.empty((n_samples, n_x))
    for i, prior_sample in enumerate(raw_prior_samples):
        x_p[i] = active_vector_from_sample(
            prior_sample, name=f"CG prior sample {i}"
        )

    samples = np.empty((n_samples, n_x))
    cg_iters = np.empty(n_samples, dtype=int)

    for i in range(n_samples):
        eps = rng.standard_normal(n_d)
        rhs = Q_prior @ x_p[i] + J_tilde_map.T @ eps

        # Initial guess from prior-preconditioner (good warm start)
        x0 = L_prior.solve(rhs)

        n_it = [0]
        def _cb(_xk):
            n_it[0] += 1

        s, info = cg(
            H_post_op, rhs, M=M_prec_op, x0=x0,
            rtol=cg_tol, maxiter=cg_maxiter, callback=_cb,
        )
        cg_iters[i] = n_it[0]
        if info != 0 and verbose:
            print(f"  [sample {i}] CG warning: info={info}, iters={n_it[0]}")

        samples[i] = x_MAP + s

    if verbose:
        print(f"  CG iters: mean={cg_iters.mean():.1f}, "
              f"min={cg_iters.min()}, max={cg_iters.max()}")

    return samples if n_samples > 1 else samples[0]


def prior_preconditioned_eigenpairs_gn(eig_tol=LAPLACE_EIG_TOL):
    """
    Solve J_tilde.T J_tilde v = lambda Q_prior v via the data-space dual.

    The returned V columns are Q_prior-orthonormal and QV = Q_prior @ V.
    """
    data_gram = np.asarray(J_tilde_map @ Q_inv_Jt_map, dtype=float)
    data_gram = 0.5 * (data_gram + data_gram.T)

    eigvals_all, U_all = np.linalg.eigh(data_gram)
    order = np.argsort(eigvals_all)[::-1]
    eigvals_all = np.maximum(eigvals_all[order], 0.0)
    U_all = U_all[:, order]

    keep_idx = np.flatnonzero(eigvals_all >= eig_tol)
    eigvals = eigvals_all[keep_idx]
    info = {
        "eigvals_all": eigvals_all,
        "n_positive": int(np.count_nonzero(eigvals_all > 0.0)),
        "n_kept": int(eigvals.size),
    }

    if eigvals.size == 0:
        empty = np.zeros((n_x, 0))
        return np.zeros(0), empty, empty, info

    U = U_all[:, keep_idx]
    inv_sqrt_lam = 1.0 / np.sqrt(eigvals)
    V = Q_inv_Jt_map @ (U * inv_sqrt_lam[None, :])
    QV = J_tilde_map.T @ (U * inv_sqrt_lam[None, :])

    return eigvals, V, QV, info


def sample_posterior_low_rank(x_map, eigvals, V, QV, prior_draws):
    """
    Sample N(x_map, H_post^{-1}) by shrinking prior draws in data-informed
    Q_prior-orthonormal directions.
    """
    perturbations = np.asarray(prior_draws, dtype=float).copy()

    if eigvals.size > 0:
        alpha = QV.T @ perturbations.T
        shrink = 1.0 / np.sqrt(1.0 + eigvals) - 1.0
        correction = V @ (shrink[:, None] * alpha)
        perturbations += correction.T

    return np.asarray(x_map, dtype=float)[None, :] + perturbations


def posterior_variance():
    """
    Compute diagonal of posterior covariance Sigma_post via Woodbury.
    diag(Sigma_post) = diag(Q_prior^{-1}) - diag(Q_prior^{-1} J_tilde^T M^{-1} J_tilde Q_prior^{-1})
    """
    # diag(Q_prior^{-1}): solve Q_prior @ e_i for each i (expensive for large n_x)
    # Use the factored form: Q_inv_Jt already computed
    # diag(A B) = sum(A * B.T, axis=1) for A @ B^T
    M_inv_Jt_Qinv = np.linalg.solve(M_post, Q_inv_Jt_map.T)  # (n_d, n_x)
    # correction_diag = diag(Q_inv_Jt_map @ M_inv_Jt_Qinv) = row-wise dot
    correction_diag = np.sum(Q_inv_Jt_map * M_inv_Jt_Qinv.T, axis=1)

    # Prior variance diagonal
    prior_var = np.array([L_prior.solve(np.eye(1, n_x, i).flatten())[i] for i in range(n_x)])

    return prior_var - correction_diag


# ============================================================
# Generate posterior samples and compute uncertainty
# ============================================================
print("\nComputing posterior variance...")
post_var = posterior_variance()
print(f"Posterior std range: [{np.sqrt(post_var.min()):.4f}, {np.sqrt(post_var.max()):.4f}]")
print(f"Prior std (log-space): {log_std:.4f}")

print("\nSolving prior-preconditioned GN eigenproblem...")
eigvals_gn, V_gn, QV_gn, eig_info_gn = prior_preconditioned_eigenpairs_gn()
eigvals_all_gn = eig_info_gn["eigvals_all"]
if eigvals_all_gn.size:
    print(f"  data-space eigenvalues: max={eigvals_all_gn[0]:.3e}, "
          f"positive={eig_info_gn['n_positive']}, "
          f"kept={eig_info_gn['n_kept']} (tol={LAPLACE_EIG_TOL:.1e})")
    if eigvals_gn.size:
        print(f"  retained lambda range: [{eigvals_gn[-1]:.3e}, "
              f"{eigvals_gn[0]:.3e}]")
else:
    print("  no data-space eigenvalues found")

print("\nDrawing posterior samples with CG precision solves...")
N_SAMPLES = 100
rng = np.random.default_rng(SEED+2)
post_samples_cg = sample_posterior(n_samples=N_SAMPLES, rng=rng, verbose=True)
print(f"CG posterior samples shape: {post_samples_cg.shape}")

print("\nDrawing posterior samples with low-rank shrinkage...")
rng = np.random.default_rng(SEED+2)
raw_prior_samples = jd.sample_from_precision(
    Q_prior, mask, n_samples=N_SAMPLES, rng=rng,
)
if not isinstance(raw_prior_samples, list):
    raw_prior_samples = [raw_prior_samples]
if len(raw_prior_samples) != N_SAMPLES:
    raise ValueError(
        f"jd.sample_from_precision returned {len(raw_prior_samples)} samples, "
        f"expected {N_SAMPLES}"
    )
prior_draws = np.empty((N_SAMPLES, n_x))
for i, prior_sample in enumerate(raw_prior_samples):
    prior_draws[i] = active_vector_from_sample(
        prior_sample, name=f"low-rank prior sample {i}"
    )

post_samples_low_rank = sample_posterior_low_rank(
    x_MAP, eigvals_gn, V_gn, QV_gn, prior_draws
)
print(f"Low-rank posterior samples shape: {post_samples_low_rank.shape}")
post_samples = post_samples_cg

print("\nComputing prediction responses without Jacobians...")
print("  Computing MAP predictions...")
d_pred_map = compute_predictions(x_MAP)

n_forward = min(10, N_SAMPLES)
print(f"  Simulating {n_forward} prior ensemble members...")
d_pred_prior_samples = []
for i in range(n_forward):
    print(f"    prior sample {i+1}/{n_forward}")
    x_prior = mu_prior + prior_draws[i]
    d_pred_prior_samples.append(compute_predictions(x_prior))
d_pred_prior_samples = np.stack(d_pred_prior_samples, axis=0)

print(f"  Simulating {n_forward} CG posterior ensemble members...")
d_pred_post_samples = []
for i in range(n_forward):
    print(f"    CG posterior sample {i+1}/{n_forward}")
    d_pred_post_samples.append(compute_predictions(post_samples_cg[i]))
d_pred_post_samples = np.stack(d_pred_post_samples, axis=0)

print(f"  Simulating {n_forward} low-rank GN posterior ensemble members...")
d_pred_post_gn = []
for i in range(n_forward):
    print(f"    low-rank GN posterior sample {i+1}/{n_forward}")
    d_pred_post_gn.append(compute_predictions(post_samples_low_rank[i]))
d_pred_post_gn = np.stack(d_pred_post_gn, axis=0)

# Save results
np.savez('laplace_results.npz',
         x_MAP=x_MAP,
         mu_prior=mu_prior,
         post_var=post_var,
         post_samples=post_samples,
         post_samples_cg=post_samples_cg,
         post_samples_low_rank=post_samples_low_rank,
         prior_draws=prior_draws,
         prior_samples=prior_draws,
         gn_eigvals=eigvals_gn,
         gn_eigvals_all=eigvals_all_gn,
         gn_eigvecs=V_gn,
         gn_Qeigvecs=QV_gn,
         gn_post_samples=post_samples_low_rank,
         laplace_eig_tol=np.array(LAPLACE_EIG_TOL),
         J_tilde_map=J_tilde_map,
         mask=mask,
         d_obs=d_obs,
         d_pred_map=d_pred_map,
         d_pred_prior_samples=d_pred_prior_samples,
         d_pred_post_samples=d_pred_post_samples,
         d_pred_post_cg=d_pred_post_samples,
         d_pred_post_low_rank=d_pred_post_gn,
         d_pred_post_gn=d_pred_post_gn)

print("\nResults saved to laplace_results.npz")
print("Done.")
