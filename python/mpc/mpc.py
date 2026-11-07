"""Python port of the SRB (single rigid body) convex MPC (mpc.cpp / mpc.h / srb_model.h).

Same structure as the C++:
    buildDiscreteModels -> buildCondensedDynamics -> buildCost -> buildConstraints -> solveQP
The QP backend is OSQP (the C++ uses a cvxpygen solver; the problem is identical).

    pip install numpy scipy osqp

    python python/mpc.py
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import osqp
import scipy.sparse as sp
from scipy.linalg import expm

# ----------------------------------------------------------------------------
# srb_model
# ----------------------------------------------------------------------------
NX = 13  # state dimension
NU = 12  # input dimension (4 feet x 3)
NUM_LEGS = 4

# Index offsets into the 13-vector state
IDX_THETA = 0  # roll, pitch, yaw
IDX_POS = 3
IDX_OMEGA = 6
IDX_VEL = 9
IDX_G = 12

CONSTRAINTS_PER_FOOT = 5  # 4 friction-pyramid rows + 1 normal-force row
PYRAMID_ROWS = 4
INF = 1e20


@dataclass
class State:
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    linear_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    g: float = -9.81

    def to_vector(self) -> np.ndarray:
        return np.concatenate((
            [self.roll, self.pitch, self.yaw],
            self.pos, self.angular_vel, self.linear_vel, [self.g],
        )).astype(float)

    @staticmethod
    def from_vector(x: np.ndarray) -> "State":
        return State(
            roll=float(x[0]), pitch=float(x[1]), yaw=float(x[2]),
            pos=np.array(x[IDX_POS:IDX_POS + 3], dtype=float),
            angular_vel=np.array(x[IDX_OMEGA:IDX_OMEGA + 3], dtype=float),
            linear_vel=np.array(x[IDX_VEL:IDX_VEL + 3], dtype=float),
            g=float(x[IDX_G]),
        )


@dataclass
class Input:
    """Ground reaction forces f1..f4, world frame; shape (4, 3)."""
    foot_ground_force: np.ndarray = field(default_factory=lambda: np.zeros((NUM_LEGS, 3)))

    def to_vector(self) -> np.ndarray:
        return self.foot_ground_force.reshape(-1).copy()

    @staticmethod
    def from_vector(u: np.ndarray) -> "Input":
        return Input(np.asarray(u, dtype=float).reshape(NUM_LEGS, 3).copy())


def skew(r: np.ndarray) -> np.ndarray:
    """skew(r) @ f == np.cross(r, f)"""
    return np.array([[0.0, -r[2], r[1]],
                     [r[2], 0.0, -r[0]],
                     [-r[1], r[0], 0.0]])


def rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0],
                     [s, c, 0.0],
                     [0.0, 0.0, 1.0]])


class Model:
    """Continuous-time model: x_dot = A x + B u"""

    def __init__(self):
        self.A = np.zeros((NX, NX))
        self.B = np.zeros((NX, NU))

    def build_A(self, yaw: float) -> None:
        A = self.A
        A[:] = 0.0
        A[IDX_THETA:IDX_THETA + 3, IDX_OMEGA:IDX_OMEGA + 3] = rot_z(yaw).T  # theta_dot = Rz^T omega
        A[IDX_POS:IDX_POS + 3, IDX_VEL:IDX_VEL + 3] = np.eye(3)              # p_dot = v
        A[IDX_VEL:IDX_VEL + 3, IDX_G] = (0.0, 0.0, 1.0)                      # v_dot += [0,0,1] g

    def build_B(self, r: np.ndarray, I_world_inv: np.ndarray, mass: float) -> None:
        """r: (4, 3) foot positions relative to CoM, world frame."""
        B = self.B
        B[:] = 0.0
        for i in range(NUM_LEGS):
            B[IDX_OMEGA:IDX_OMEGA + 3, 3 * i:3 * i + 3] = I_world_inv @ skew(r[i])
            B[IDX_VEL:IDX_VEL + 3, 3 * i:3 * i + 3] = np.eye(3) / mass

    @staticmethod
    def world_inertia_inv(I_body: np.ndarray, yaw: float) -> np.ndarray:
        R = rot_z(yaw)
        return np.linalg.inv(R @ I_body @ R.T)


@dataclass
class DiscreteModel:
    Ad: np.ndarray
    Bd: np.ndarray

    @staticmethod
    def discretize(m: Model, dt: float) -> "DiscreteModel":
        return DiscreteModel(Ad=expm(m.A * dt), Bd=m.B * dt)  # B: first-order approximation


# ----------------------------------------------------------------------------
# mpc
# ----------------------------------------------------------------------------
@dataclass
class MPCProblem:
    x0: State                       # measured current state
    x_ref: List[np.ndarray]         # N vectors (13,), desired state at steps 1..N; gravity slot == State.g
    foot_rel_pos: List[np.ndarray]  # N arrays (4, 3): foot position minus CoM, world frame
    contact: List[List[bool]]       # N lists of 4 bools: True = stance, False = swing


@dataclass
class MPCResult:
    success: bool = False
    u0: Input = field(default_factory=Input)   # forces to apply now
    U: np.ndarray = field(default_factory=lambda: np.zeros(0))  # stacked solution, size 12*N
    solve_time_ms: float = 0.0


@dataclass
class MPCConfig:
    horizon: int = 10
    dt: float = 0.03
    mass: float = 12.0
    I_body: np.ndarray = field(default_factory=lambda: np.eye(3))
    mu: float = 0.4
    f_min: float = 0.0
    f_max: float = 150.0
    # [roll pitch yaw | x y z | wx wy wz | vx vy vz | g]
    Q_diag: np.ndarray = field(default_factory=lambda: np.ones(NX))
    R_weight: float = 1e-4


class MPC:
    def __init__(self, cfg: Optional[MPCConfig] = None):
        self.cfg = cfg or MPCConfig()
        N = self.N = self.cfg.horizon

        self.models: List[DiscreteModel] = []

        self.A_qp = np.zeros((NX * N, NX))
        self.B_qp = np.zeros((NX * N, NU * N))  # upper block triangle stays zero
        self.H = np.zeros((NU * N, NU * N))
        self.q = np.zeros(NU * N)

        n_con = CONSTRAINTS_PER_FOOT * NUM_LEGS * N
        self.C = np.zeros((n_con, NU * N))
        self.lb = np.zeros(n_con)
        self.ub = np.zeros(n_con)

        mu = self.cfg.mu
        C_foot = np.array([[1, 0, -mu],
                           [-1, 0, -mu],
                           [0, 1, -mu],
                           [0, -1, -mu],
                           [0, 0, 1]], dtype=float)

        for i in range(N):
            for j in range(NUM_LEGS):
                row = CONSTRAINTS_PER_FOOT * (NUM_LEGS * i + j)
                col = NU * i + 3 * j
                self.C[row:row + CONSTRAINTS_PER_FOOT, col:col + 3] = C_foot
                self.lb[row:row + PYRAMID_ROWS] = -INF
                self.ub[row:row + PYRAMID_ROWS] = 0.0

        # constant weights
        self.L_diag = np.tile(np.asarray(self.cfg.Q_diag, dtype=float), N)
        self.K_diag = np.full(NU * N, self.cfg.R_weight)

        self.U_sol = np.zeros(NU * N)
        self.U_prev = np.zeros(NU * N)

        # OSQP state (one-time setup)
        self._C_sp = sp.csc_matrix(self.C)
        self._triu_idx = np.triu_indices(NU * N)  # fixed sparsity pattern for P (upper triangle)
        self._P_pattern = sp.csc_matrix(
            (np.ones(len(self._triu_idx[0])), (self._triu_idx[0], self._triu_idx[1])),
            shape=(NU * N, NU * N))
        # csc data order differs from the raw triu_indices order -> map it once
        self._P_order = self._csc_order()
        self.solver_initialized = False
        self._solver = None

    def _csc_order(self) -> np.ndarray:
        """Indices s.t. H[triu_idx][order] matches the csc data layout of the pattern."""
        n = self.N * NU
        tag = sp.csc_matrix(
            (np.arange(1, len(self._triu_idx[0]) + 1, dtype=float), (self._triu_idx[0], self._triu_idx[1])),
            shape=(n, n))
        return tag.data.astype(int) - 1  # +1/-1: keeps index 0 from being an explicit zero

    # -- 1 -------------------------------------------------------------------
    def build_discrete_models(self, p: MPCProblem) -> None:
        self.models = []
        for i in range(self.cfg.horizon):
            yaw = float(p.x_ref[i][2])
            m = Model()
            m.build_A(yaw)
            m.build_B(np.asarray(p.foot_rel_pos[i]), Model.world_inertia_inv(self.cfg.I_body, yaw), self.cfg.mass)
            self.models.append(DiscreteModel.discretize(m, self.cfg.dt))

    # -- 2: X = A_qp x0 + B_qp U --------------------------------------------
    def build_condensed_dynamics(self) -> None:
        N = self.N
        A, B = self.A_qp, self.B_qp
        for i in range(N):
            Ad, Bd = self.models[i].Ad, self.models[i].Bd
            rs = slice(NX * i, NX * (i + 1))
            if i == 0:
                A[rs, :] = Ad
            else:
                A[rs, :] = Ad @ A[NX * (i - 1):NX * i, :]
            B[rs, NU * i:NU * (i + 1)] = Bd
            for j in range(i):
                B[rs, NU * j:NU * (j + 1)] = Ad @ B[NX * (i - 1):NX * i, NU * j:NU * (j + 1)]

    # -- 3: min 0.5 U'HU + q'U ----------------------------------------------
    def _error(self, p: MPCProblem) -> np.ndarray:
        Y = np.concatenate([np.asarray(x, dtype=float) for x in p.x_ref])
        return self.A_qp @ p.x0.to_vector() - Y

    def build_cost(self, p: MPCProblem) -> None:
        err = self._error(p)
        BtL = self.B_qp.T * self.L_diag  # B' diag(L)
        self.q = 2.0 * BtL @ err
        self.H = 2.0 * BtL @ self.B_qp
        self.H[np.diag_indices_from(self.H)] += 2.0 * self.K_diag

    # -- 4: lb <= C U <= ub --------------------------------------------------
    def build_constraints(self, p: MPCProblem) -> None:
        for i in range(self.N):
            for j in range(NUM_LEGS):
                r = CONSTRAINTS_PER_FOOT * (NUM_LEGS * i + j) + PYRAMID_ROWS
                if p.contact[i][j]:
                    self.lb[r], self.ub[r] = self.cfg.f_min, self.cfg.f_max
                else:  # swing: fz pinned to 0 (pyramid then forces fx = fy = 0)
                    self.lb[r], self.ub[r] = 0.0, 0.0

    # -- 5 -------------------------------------------------------------------
    def solve_qp(self, p: MPCProblem) -> Optional[np.ndarray]:
        Px = self.H[self._triu_idx][self._P_order]
        if not self.solver_initialized:
            P = self._P_pattern.copy()
            P.data = Px
            self._solver = osqp.OSQP()
            self._solver.setup(P=P, q=self.q, A=self._C_sp, l=self.lb, u=self.ub,
                               verbose=False, warm_start=True,
                               eps_abs=1e-4, eps_rel=1e-4, max_iter=4000, polish=False)
            self.solver_initialized = True
        else:
            self._solver.update(Px=Px, q=self.q, l=self.lb, u=self.ub)
        self._solver.warm_start(x=self.shift(self.U_prev))

        res = self._solver.solve()
        if res.info.status != "Solved" and res.info.status != "solved":
            return None
        return np.array(res.x)

    def shift(self, U: np.ndarray) -> np.ndarray:
        """[u0 u1 ... uN-1] -> [u1 ... uN-1 uN-1]"""
        out = np.empty_like(U)
        out[:NU * (self.N - 1)] = U[NU:]
        out[NU * (self.N - 1):] = U[NU * (self.N - 1):]
        return out

    def solve(self, p: MPCProblem) -> MPCResult:
        t0 = time.perf_counter()

        self.build_discrete_models(p)
        self.build_condensed_dynamics()
        self.build_cost(p)
        self.build_constraints(p)

        sol = self.solve_qp(p)
        res = MPCResult(success=sol is not None)
        self.U_sol = sol if sol is not None else self.shift(self.U_prev)

        res.U = self.U_sol.copy()
        res.u0 = Input.from_vector(self.U_sol[:NU])
        self.U_prev = self.U_sol.copy()
        res.solve_time_ms = (time.perf_counter() - t0) * 1000.0
        return res


# ----------------------------------------------------------------------------
# demo: robot standing still on four feet
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = MPCConfig(horizon=10, dt=0.03, mass=12.0,
                    I_body=np.diag([0.07, 0.26, 0.242]),
                    Q_diag=np.array([5, 5, 10, 1, 1, 50, 1, 1, 1, 1, 1, 1, 0.0]))
    mpc = MPC(cfg)

    x0 = State(pos=np.array([0, 0, 0.3]))
    x_ref = x0.to_vector()
    feet = np.array([[0.2, 0.1, -0.3], [0.2, -0.1, -0.3], [-0.2, 0.1, -0.3], [-0.2, -0.1, -0.3]])

    prob = MPCProblem(
        x0=x0,
        x_ref=[x_ref.copy() for _ in range(cfg.horizon)],
        foot_rel_pos=[feet for _ in range(cfg.horizon)],
        contact=[[True] * NUM_LEGS for _ in range(cfg.horizon)],
    )
    r = mpc.solve(prob)
    print("success:", r.success, f"({r.solve_time_ms:.2f} ms)")
    print("foot forces:\n", np.round(r.u0.foot_ground_force, 2))
    print("sum fz:", r.u0.foot_ground_force[:, 2].sum(), "expected ~", 12 * 9.81)

    # trot-like: swing diagonal pair
    prob.contact = [[True, False, False, True] for _ in range(cfg.horizon)]
    r = mpc.solve(prob)
    print("\ntrot stance:", r.success, f"({r.solve_time_ms:.2f} ms)")
    print(np.round(r.u0.foot_ground_force, 2))