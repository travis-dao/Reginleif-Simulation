"""
mpc_qp.py  --  standalone Python QP solver for the condensed MPC (no C codegen).

pip install cvxpy numpy
python python/mpc_qp.py            # sanity-check against the explicit dense-H formulation

Usage:
    qp  = MPCQPSolver(Config(horizon=10))
    res = qp.solved(B_qp, err, lb_f, ub_f)
    U   = res.U            # shape (NU*N,)
"""
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import cvxpy as cp

NX = 13
NU = 12
NUM_LEGS = 4
PYRAMID_ROWS = 4

# Default OSQP options (override per call via solved(..., max_iter=..., ...))
OSQP_OPTS = {
    "max_iter": 4000,
    "eps_abs": 1e-4,
    "eps_rel": 1e-4,
    "polishing": False,
}


@dataclass
class Config:
    horizon: int = 10
    mu: float = 0.6
    R_weight: float = 1e-5
    # per-state weights, order must match your StateVec
    Q_diag: np.ndarray = field(default_factory=lambda: np.array(
        [100, 100, 10,      # roll pitch yaw
         10, 10, 100,       # x y z
         1, 1, 1,           # omega
         1, 1, 1,           # vel
         0.0]))             # gravity state


@dataclass
class QPResult:
    U: Optional[np.ndarray]   # optimal stacked inputs (NU*N,), None if solve failed
    status: str               # cvxpy/OSQP status string
    cost: Optional[float]     # optimal objective value
    solve_time: float         # wall-clock seconds spent in solve

    @property
    def ok(self) -> bool:
        return self.U is not None and self.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)


# ----------------------------------------------------------------------------
# constant matrices
# ----------------------------------------------------------------------------
def friction_pyramid(cfg: Config) -> np.ndarray:
    """Constant block: C_pyr @ U <= 0, shape (4*NUM_LEGS*N, NU*N)."""
    mu, N = cfg.mu, cfg.horizon
    C_foot = np.array([[ 1,  0, -mu],
                       [-1,  0, -mu],
                       [ 0,  1, -mu],
                       [ 0, -1, -mu]], dtype=float)
    C = np.zeros((PYRAMID_ROWS * NUM_LEGS * N, NU * N))
    for i in range(N):
        for j in range(NUM_LEGS):
            row = PYRAMID_ROWS * (NUM_LEGS * i + j)
            col = NU * i + 3 * j
            C[row:row + PYRAMID_ROWS, col:col + 3] = C_foot
    return C


def fz_selector(cfg: Config) -> np.ndarray:
    """Picks fz of every (step, foot): shape (NUM_LEGS*N, NU*N)."""
    N = cfg.horizon
    S = np.zeros((NUM_LEGS * N, NU * N))
    for i in range(N):
        for j in range(NUM_LEGS):
            S[NUM_LEGS * i + j, NU * i + 3 * j + 2] = 1.0
    return S


# ----------------------------------------------------------------------------
# solver class
# ----------------------------------------------------------------------------
class MPCQPSolver:
    """Condensed MPC QP:

        min_U  || sqrt(Q) (B_qp U + err) ||^2 + R ||U||^2
        s.t.   C_pyr U <= 0                (friction pyramid)
               lb_f <= Fz U <= ub_f        (fz bounds, 0/0 for swing legs)

    The cvxpy problem is built once; each call to `solved` only updates the
    parameter values, so cvxpy re-uses the cached canonicalization (DPP) and
    warm-starts OSQP from the previous solution.
    """

    def __init__(self, cfg: Optional[Config] = None, solver_opts: Optional[dict] = None):
        self.cfg = cfg or Config()
        self.N = self.cfg.horizon
        self.solver_opts = dict(OSQP_OPTS if solver_opts is None else solver_opts)
        self.prob, self._vars = self._build_problem()

    def _build_problem(self):
        cfg, N = self.cfg, self.N

        # parameters (updated every MPC tick)
        B_qp = cp.Parameter((NX * N, NU * N), name="B_qp")   # condensed input matrix
        err  = cp.Parameter(NX * N,           name="err")    # A_qp @ x0 - Y
        lb_f = cp.Parameter(NUM_LEGS * N,     name="lb_f")   # fz lower bound (f_min or 0)
        ub_f = cp.Parameter(NUM_LEGS * N,     name="ub_f")   # fz upper bound (f_max or 0)

        # constants
        L_sqrt = np.sqrt(np.tile(cfg.Q_diag, N))
        C_pyr = friction_pyramid(cfg)
        Fz = fz_selector(cfg)

        # variable
        U = cp.Variable(NU * N, name="U")

        cost = (cp.sum_squares(cp.multiply(L_sqrt, B_qp @ U + err))
                + cfg.R_weight * cp.sum_squares(U))

        constraints = [
            C_pyr @ U <= 0,
            Fz @ U >= lb_f,
            Fz @ U <= ub_f,
        ]

        prob = cp.Problem(cp.Minimize(cost), constraints)
        assert prob.is_dcp(dpp=True), "problem is not DPP"
        return prob, dict(U=U, B_qp=B_qp, err=err, lb_f=lb_f, ub_f=ub_f)

    def solved(self, B_qp, err, lb_f, ub_f, warm_start: bool = True,
               **solver_overrides) -> QPResult:
        """Solve the QP for the given parameters.

        Args:
            B_qp:  (NX*N, NU*N)  condensed input matrix
            err:   (NX*N,)       A_qp @ x0 - Y
            lb_f:  (NUM_LEGS*N,) fz lower bounds (f_min for stance, 0 for swing)
            ub_f:  (NUM_LEGS*N,) fz upper bounds (f_max for stance, 0 for swing)
            warm_start: warm-start OSQP from the previous solution
            **solver_overrides: any OSQP option (e.g. max_iter=200, eps_abs=1e-6)

        Returns:
            QPResult with .U, .status, .cost, .solve_time, .ok
        """
        N = self.N
        B_qp = np.asarray(B_qp, dtype=float)
        err  = np.asarray(err,  dtype=float).reshape(-1)
        lb_f = np.asarray(lb_f, dtype=float).reshape(-1)
        ub_f = np.asarray(ub_f, dtype=float).reshape(-1)

        expected = {
            "B_qp": ((NX * N, NU * N), B_qp.shape),
            "err":  ((NX * N,),        err.shape),
            "lb_f": ((NUM_LEGS * N,),  lb_f.shape),
            "ub_f": ((NUM_LEGS * N,),  ub_f.shape),
        }
        for name, (want, got) in expected.items():
            if want != got:
                raise ValueError(f"{name} has shape {got}, expected {want}")

        v = self._vars
        v["B_qp"].value = B_qp
        v["err"].value = err
        v["lb_f"].value = lb_f
        v["ub_f"].value = ub_f

        opts = {**self.solver_opts, **solver_overrides}
        t0 = time.perf_counter()
        try:
            self.prob.solve(solver=cp.OSQP, warm_start=warm_start, **opts)
        except cp.error.SolverError:
            return QPResult(None, "solver_error", None, time.perf_counter() - t0)
        dt = time.perf_counter() - t0

        U = None if v["U"].value is None else v["U"].value.copy()
        return QPResult(U, self.prob.status, self.prob.value, dt)

    # convenience alias
    solve = solved


# ----------------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------------
def test(cfg: Optional[Config] = None):
    """Compare solver output against the explicit dense-H formulation."""
    cfg = cfg or Config()
    N = cfg.horizon
    qp = MPCQPSolver(cfg)
    rng = np.random.default_rng(0)

    # random lower-block-triangular B
    B = np.zeros((NX * N, NU * N))
    for i in range(N):
        for j in range(i + 1):
            B[NX*i:NX*(i+1), NU*j:NU*(j+1)] = 0.01 * rng.standard_normal((NX, NU))
    e = rng.standard_normal(NX * N)

    contact = rng.random((N, NUM_LEGS)) > 0.4
    f_min, f_max = 5.0, 150.0
    lb = np.where(contact, f_min, 0.0).reshape(-1)
    ub = np.where(contact, f_max, 0.0).reshape(-1)

    # tight tolerances so the comparison is meaningful
    res = qp.solved(B, e, lb, ub, warm_start=False,
                    max_iter=20000, eps_abs=1e-6, eps_rel=1e-6)

    # reference: same math as buildCost (H, q) with explicit constraints
    L = np.tile(cfg.Q_diag, N)
    H = 2 * B.T @ (L[:, None] * B) + 2 * cfg.R_weight * np.eye(NU * N)
    q = 2 * B.T @ (L * e)
    U = cp.Variable(NU * N)
    ref = cp.Problem(cp.Minimize(0.5 * cp.quad_form(U, cp.psd_wrap(H)) + q @ U),
                     [friction_pyramid(cfg) @ U <= 0,
                      fz_selector(cfg) @ U >= lb,
                      fz_selector(cfg) @ U <= ub])
    ref.solve(solver=cp.OSQP, eps_abs=1e-6, eps_rel=1e-6)

    print("status:", res.status, "| ok:", res.ok,
          "| solve_time: %.2f ms" % (1e3 * res.solve_time),
          "| max |U - U_ref| =", np.abs(res.U - U.value).max())


if __name__ == "__main__":
    test()