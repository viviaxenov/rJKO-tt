import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import expm
from scipy.interpolate import RegularGridInterpolator


import tt
import tt.amen
import tt.multifuncrs
import tt.cross.rectcross

from functools import cache
from typing import Literal, Callable, Tuple, List
import traceback

from utility import *

np.seterr(all="raise")

# TODO: make convenient submodule like rcParams in Matplotlib
tt_precision = 1e-13
density_min = tt_precision
nonlinear_tol = 1e-6
global_max_rk = 20

# TODO: implement in Chebyshev basis
# (or any other more meaningful basis)
# (it may imporve N and also allow for continous interpretation, for example, to use Galerkin approximation of the terminal conditio)
# IDK how to enforce Neumann boundary condition eta_n |dX = 0 in Chebyshev basis
# there is some information in Oseledts paper
# and ref. [Trefethen 2000] there
# in principle we have 2 last rows zero in the matrix below, maybe we can write the BC control down there
# def get_dxx_matrix(N):
#     basis = np.eye(N)
#     lap = chbp.chebder(basis, m=2, axis=0)
#
#     return np.vstack((lap, np.zeros((2, N))))
#
# def get_heat_equation_matrix(N, beta, T):
#     N += 1
#     # operator that describes the solution of
#     #  u_t =  \beta \Delta{u} when time changes by T
#
#     lap = get_dxx_matrix(N)
#
#     return expm(beta*T*lap)


@cache
def get_dxx_matrix(N, h_x):
    sub_diag = np.full((N - 1,), 1.0)
    main_diag = np.concatenate(
        (
            (-1.0,),
            np.full((N - 2,), -2.0),
            (-1.0,),
        )
    )

    Lap = (
        diags(
            [sub_diag, main_diag, sub_diag],
            offsets=(-1, 0, 1),
        )
        / h_x**2
    )

    return Lap


# @cache
def get_heat_equation_matrix(N, h_x, beta, T):
    Lap = get_dxx_matrix(N, h_x)
    expLap = expm(beta * T * Lap)
    return expLap.todense()


def solve_heat_TT(
    eta: tt.vector, beta: np.float64, T: np.float64, N: int, h_x: np.float64
):
    U = get_heat_equation_matrix(N, h_x, beta, T)

    return tt.vector.from_list(
        [np.einsum("ij,kjl->kil", U, _v) for _v in tt.vector.to_list(eta)]
    )


# TODO make a part of solve_initial_condition
def solve_hadamard_TT(eta: tt.vector, rho_0: tt.vector, hat_eta_initial=None):
    """
    Find \hat\eta such that \hat\eta(x)\eta(x) = \rho_0(x)
    """
    eta_as_oper = tt.matrix.from_list(
        [
            np.einsum("ij,kil->kijl", np.eye(_v.shape[1]), _v)
            for _v in tt.vector.to_list(eta)
        ]
    )
    # TODO what should  be the initial iterate?
    # need to keep this guy from previous Fixed-Point iterate
    if hat_eta_initial is None:
        # hat_eta_initial = tt.vector.from_list([np.full((1, _v.shape[1], 1), 1.) for _v in tt.vector.to_list(eta)])
        hat_eta_initial = rho_0

    hat_eta = tt.amen.amen_solve(
        eta_as_oper, rho_0, hat_eta_initial, 1e-10, verb=0, local_iters=10
    )

    return hat_eta


def solve_initial_condition(
    eta: tt.vector,
    rho_0: tt.vector,
    method: Literal["amen", "cross", "full"] = "cross",
    hat_eta_initial: tt.vector = None,
):
    print(f"\t\tSolving initial {method=}", flush=True)

    if method not in ["cross", "full"]:
        raise NotImplemented

    elif method == "cross":
        hat_eta = tt.multifuncrs2(
            [rho_0, eta],
            lambda _x: _x[:, 0, ...] / _x[:, 1, ...],
            y0=hat_eta_initial,
            verb=0,
            eps=nonlinear_tol,
            kickrank=2,
        )
        print("\t\tRounding", flush=True)
        hat_eta = hat_eta.round(eps=tt_precision, rmax=global_max_rk)

    elif method == "full":
        hat_eta = tt.vector(
            rho_0.full() / eta.full(),
            eps=tt_precision,
            rmax=global_max_rk,
        )

    return hat_eta


def solve_terminal_condition(
    hat_eta: tt.vector,
    rho_infty_on_grid: Callable,
    beta: np.float64,
    method: Literal["cross", "full"] = "cross",
    eta_initial: tt.vector = None,
):
    print(f"\t\tSolving terminal, {method=}", flush=True)
    if method not in ["cross", "full"]:
        raise NotImplemented

    elif method == "cross":
        if eta_initial is None:
            eta_initial = tt.rand(hat_eta.n, r=hat_eta.r)
        eta = tt.cross.rectcross.cross(
            lambda _I: (rho_infty_on_grid(_I) / np.maximum(tt_take(hat_eta, _I), density_min))
            ** (1.0 / (1.0 + 2.0 * beta)),
            eta_initial,
            eps=nonlinear_tol,
            kickrank=2,
            verbose=False,
        )

        print("\t\tRounding", flush=True)
        eta = eta.round(eps=tt_precision, rmax=global_max_rk)

    else:
        eta = tt.vector(
            ((rho_infty_on_grid) / (np.maximum(hat_eta.full(), density_min)))
            ** (1.0 / (1.0 + 2.0 * beta)),
            eps=tt_precision,
        )

    # print(eta)
    # print("Solving terminal done")
    return eta


def fixed_point_inner_cycle(
    eta: tt.vector,
    rho_0: tt.vector,
    rho_infty: Union[Callable, np.ndarray],
    grid: np.ndarray,
    beta: np.float64,
    T: np.float64,
    initial_method="cross",
    terminal_method="cross",
    start_value_init: tt.vector = None,
    start_value_term: tt.vector = None,
):
    # Assuming that the whole grid is a Cartesian product of d identical uniform grids
    #
    N = grid.shape[0]
    h_x = grid[1] - grid[0]

    eta_t0 = solve_heat_TT(eta, -beta, -T, N, h_x)  # (4.5.2)

    hat_eta_t0 = solve_initial_condition(
        eta_t0,
        rho_0,
        method=initial_method,
        hat_eta_initial=start_value_init,
    )  # (4.5.3)
    # TODO: A. proper linear solver
    #       B. get inital value from previous step

    hat_eta_next = solve_heat_TT(hat_eta_t0, beta, T, N, h_x)  # (4.5.1)
    eta_next = solve_terminal_condition(
        hat_eta_next,
        rho_infty,
        beta,
        method=terminal_method,
        eta_initial=start_value_term,
    )

    return eta_next, hat_eta_next, eta_t0, hat_eta_t0


def fixed_point_picard(
    x_cur: tt.vector,
    g_cur: tt.vector,
    *args,
    relaxation: np.float64 = 1.0,
    max_rank: int = global_max_rk,
) -> Tuple[tt.vector, np.float64]:

    x_new = tt.vector.round(
        (1.0 - relaxation) * x_cur + relaxation * g_cur,
        rmax=max_rank,
        eps=tt_precision,
    )

    return x_new, relaxation


def fixed_point_aitken(
    x_cur: tt.vector,
    g_cur: tt.vector,
    x_prev: tt.vector,
    g_prev: tt.vector,
    relaxation: np.float64 = 1.0,
    max_rank: int = 10,
) -> Tuple[tt.vector, np.float64]:

    f_prev = g_prev - x_prev
    f_cur = g_cur - x_cur

    nom = tt.dot(f_prev, f_cur - f_prev)
    denom = tt.vector.norm(f_prev - f_cur) ** 2

    relaxation_new = -relaxation * nom / denom if denom >= 1e-16 else relaxation
    relaxation_new = np.minimum(1.0, relaxation_new)

    x_new = tt.vector.round(
        (1.0 - relaxation_new) * x_cur + relaxation_new * g_cur,
        rmax=max_rank,
        eps=tt_precision,
    )

    return x_new, relaxation_new


def init_step(
    rho_0: tt.vector, eta: tt.vector, hat_eta: tt.vector
) -> Tuple[tt.vector, tt.vector, tt.vector,]:

    if rho_0 is None:
        if eta is None or hat_eta is None:
            raise ValueError("Specify initial conditions")
        else:
            rho_0 = tt_hadamard_product(eta, hat_eta)

    else:
        if eta is None and hat_eta is None:
            eta = tt.multifuncrs2([rho_0], lambda _x: np.sqrt(_x), verb=False)

        elif eta is None:
            eta = tt.multifuncrs2(
                [rho_0, hat_eta], lambda _x: _x[:, 0] / _x[:, 1], verb=False
            )
        elif hat_eta is None:
            hat_eta = tt.multifuncrs2(
                [rho_0, eta], lambda _x: _x[:, 0] / _x[:, 1], verb=False
            )

        # else:
        #     assert (
        #         tt.vector.norm(tt_hadamard_product(eta, hat_eta) - rho_0)/tt.vector.norm(rho_0) < 1e-8
        #     ), "Initial values don't match; eta*hat_eta != rho_0"

    return rho_0, eta, hat_eta


def regularized_JKO_step(
    rho_infty: Callable,
    grid: np.ndarray,  # Unidimensional
    beta: np.float64,
    T: np.float64,
    terminal_method: Literal["cross", "full"] = "cross",
    fp_method: Literal["picard", "aitken"] = "picard",
    fp_relaxation: np.float64 = 1.,
    fp_max_rank: int = global_max_rk,
    fp_max_iter: int = 100,
    fp_stopping_rtol: np.float64 = 1e-4,
    etas_init: Tuple[tt.vector, tt.vector] = None,
    rho_0: tt.vector = None,
    debug=False,
) -> Tuple[tt.vector, tt.vector]:

    dim = rho_0.d
    h_x = grid[1] - grid[0]
    N = grid.shape[0]

    rho_0, eta_init, hat_eta_init = init_step(
        rho_0,
        *etas_init,
    )

    start_value_init = hat_eta_init
    start_value_terminal = eta_init

    fixed_point_update = (
        fixed_point_aitken if fp_method == "aitken" else fixed_point_picard
    )

    rho_inf_on_grid = (
        rho_infty(np.stack(np.meshgrid(*(grid,) * dim), axis=-1))
        if terminal_method == "full"
        else get_grid_function(rho_infty, grid)
    )

    eta_cur = eta_init
    try:
        print("\tInitializing FP")
        tilde_eta_cur, hat_eta_cur, eta_t0, hat_eta_t0 = fixed_point_inner_cycle(
            eta_cur,
            rho_0,
            rho_inf_on_grid,
            grid,
            beta,
            T,
            terminal_method=terminal_method,
            start_value_init=start_value_init,
            start_value_term=start_value_terminal,
        )
    except (RuntimeWarning, RuntimeError, FloatingPointError) as e:
        print(
            "".join(traceback.TracebackException.from_exception(e).format()),
            flush=True,
        )

        if debug:
            raise RuntimeError(
                eta_cur, hat_eta_cur, abs_errors, rel_errors, relaxations, ranks
            )
        else:
            raise RuntimeError(eta_cur, hat_eta_cur)

    eta_prev = eta_cur
    tilde_eta_prev = tilde_eta_cur

    rho_norm = tt.sum(rho_0) * h_x**dim
    abs_errors = []
    rel_errors = []
    relaxations = []
    ranks = []

    for _i in range(fp_max_iter):
        print(f"\tStarting FP step {_i + 1}", flush=True)
        try:
            tilde_eta_cur, hat_eta_cur, eta_t0, hat_eta_t0 = fixed_point_inner_cycle(
                eta_cur,
                rho_0,
                rho_inf_on_grid,
                grid,
                beta,
                T,
                terminal_method=terminal_method,
                start_value_init=start_value_init,
                start_value_term=start_value_terminal,
            )
        except (RuntimeWarning, RuntimeError, FloatingPointError) as e:
            print(
                "".join(traceback.TracebackException.from_exception(e).format()),
                flush=True,
            )

            if debug:
                raise RuntimeError(
                    eta_cur, hat_eta_cur, abs_errors, rel_errors, relaxations, ranks
                )
            else:
                raise RuntimeError(eta_cur, hat_eta_cur)

        start_value_init = hat_eta_t0
        start_value_term = eta_cur

        abs_err = tt.vector.norm(tilde_eta_cur - eta_cur)
        rel_err = abs_err / tt.vector.norm(eta_cur)
        rk = np.max(eta_cur.r)

        abs_errors.append(abs_err)
        rel_errors.append(rel_err)
        relaxations.append(fp_relaxation)
        ranks.append(rk)

        if rel_err <= fp_stopping_rtol:
            break

        eta_next, fp_relaxation = fixed_point_update(
            eta_cur,
            tilde_eta_cur,
            eta_prev,
            tilde_eta_prev,
            relaxation=fp_relaxation,
            max_rank=fp_max_rank,
        )

        eta_prev = eta_cur
        tilde_eta_prev = tilde_eta_cur

        eta_cur = eta_next

    # TODO: this is very lazy
    res_str = f"n_iterations = {_i:-4d}; {T=:-2.04f} {beta=:.04f} rk={ranks[-1]:-2d} "

    if debug:
        return (
            eta_cur,
            hat_eta_cur,
            eta_t0,
            hat_eta_t0,
            abs_errors,
            rel_errors,
            relaxations,
            ranks,
        )
    else:
        return eta_cur, hat_eta_cur, eta_t0, hat_eta_t0, res_str


# TODO do w.o. full
def get_vector_field_function(
    eta: tt.vector, hat_eta: tt.vector, grid: np.ndarray, beta: np.float64
):
    h_x = grid[1] - grid[0]
    dim = eta.d
    _grid = np.pad(grid, (1, 1), constant_values=(grid[0] - h_x, grid[-1] + h_x))
    potential_values = np.log(eta.full()) - np.log(hat_eta.full())
    potential_values = np.pad(potential_values, (1, 1), mode="edge")

    potential = RegularGridInterpolator(
        (_grid,) * dim,
        potential_values,
        bounds_error=False,
        fill_value=None,
    )

    shift = np.eye(dim) * h_x

    def _v(x: np.ndarray):
        """
        Last dimension should be dim
        """
        xp = x[..., np.newaxis, :] + shift  # xp_ijk = x_ki + h_x*delta_ij
        xm = x[..., np.newaxis, :] - shift

        grad = (potential(xp) - potential(xm)) / 2.0 / h_x

        return grad * beta

    return _v


def calculate_vector_fields(
    eta_t0,
    hat_eta_t0,
    eta_t1,
    hat_eta_t1,
    grid,
    beta,
    T,
    nsplits=0,
):
    N = grid.shape[0]
    h_x = grid[1] - grid[0]
    eta_half = solve_heat_TT(eta_t1, -beta, -T / 2.0, N, h_x)
    hat_eta_half = solve_heat_TT(hat_eta_t0, beta, T / 2.0, N, h_x)

    if nsplits == 0:
        v0 = get_vector_field_function(eta_t0, hat_eta_t0, grid, beta)
        v1 = get_vector_field_function(eta_t1, hat_eta_t1, grid, beta)

        # for RK45
        v05 = get_vector_field_function(eta_half, hat_eta_half, grid, beta)
        return [(v0, v05, v1, T)]

    return calculate_vector_fields(
        eta_t0,
        hat_eta_t0,
        eta_half,
        hat_eta_half,
        grid,
        beta,
        T / 2.0,
        nsplits=nsplits - 1,
    ) + calculate_vector_fields(
        eta_half,
        hat_eta_half,
        eta_t1,
        hat_eta_t1,
        grid,
        beta,
        T / 2.0,
        nsplits=nsplits - 1,
    )


# TODO: not the best name exactly
def gradient_flow(
    rho_0: tt.vector,
    rho_infty: Callable,
    grid: np.ndarray,  # Unidimensional
    Ts: List[np.float64],
    betas: List[np.float64],
    stopping_rtol=1e-4,
    ode_T_max=None,
    ode_time_splits=None,
):

    assert len(Ts) == len(betas)
    dim = rho_0.d
    h_x = grid[1] - grid[0]
    N = grid.shape[0]

    eta_cur = None
    hat_eta_cur = None
    rho_cur = rho_0.copy()

    rhos = [rho_0]
    rho_inf_on_grid = get_grid_function(rho_infty, grid)
    dKL = [div_KL(rho_cur, rho_inf_on_grid, h_x)]
    dKL_est = [
        dKL[-1]
    ]  # could do it, but nice initialization of eta, hat_eta is inside initial/terminal, so who cares about 1st one
    dL2 = [div_L2(rho_cur, rho_inf_on_grid, h_x)]
    rL2 = [dL2[0] / tt.vector.norm(rho_cur)]
    vector_fields = []

    stop = False
    for _iter, T in enumerate(Ts):
        beta = betas[_iter]

        try:
            print(f"Starting GF step {_iter + 1}", flush=True)
            eta_cur, hat_eta_cur, eta_t0, hat_eta_t0, res_str = regularized_JKO_step(
                rho_infty,
                grid,
                beta,
                T,
                etas_init=(eta_cur, hat_eta_cur),
                rho_0=rho_cur,
            )
        except RuntimeError as e:
            eta_cur, hat_eta_cur = e.args
            stop = True

        rho_cur = tt_hadamard_product(eta_cur, hat_eta_cur)
        rho_cur *= 1.0 / (tt.sum(rho_cur) * h_x**dim)

        dKL.append(div_KL(rho_cur, rho_inf_on_grid, h_x))
        dL2.append(div_L2(rho_cur, rho_inf_on_grid, h_x))
        rL2.append(dL2[-1] / tt.vector.norm(rho_cur))
        dKL_est.append(KL_stupid(eta_cur, hat_eta_cur, beta, h_x))

        print(
            f"{res_str} dKL={dKL[-1]:.2e} dL2={dL2[-1]:.2e} rel. err L2={rL2[-1]:.2e}"
        )

        # rhos.append(rho_cur)
        if ode_T_max is not None and T > ode_T_max:
            ode_time_splits = int(np.ceil(np.log2(T / ode_T_max)))

        if ode_time_splits is not None:
            vector_fields += calculate_vector_fields(
                eta_t0, hat_eta_t0, eta_cur, hat_eta_cur, grid, beta, T, ode_time_splits
            )

        stop = stop or (rL2[-1] <= stopping_rtol) or (rL2[-1] > 2.0 * rL2[-2])
        if stop:
            break

    rhos.append(rho_cur)

    err_stats = {
        "KL": dKL,
        "L2": dL2,
        "rel.L2": rL2,
        "KL_est": dKL_est,
    }

    if ode_time_splits:
        return rhos, err_stats, vector_fields
    return rhos, err_stats


def _ode_step_explicit_Euler(
    xs: np.ndarray,
    vector_fields: Tuple[Callable],
) -> np.ndarray:
    v0, _, _, T = vector_fields
    return xs + T * v0(xs)


def _ode_step_RK45(
    xs: np.ndarray,
    vector_fields: Tuple[Callable],
) -> np.ndarray:
    v0, v05, v1, T = vector_fields

    k1 = v0(xs)
    k2 = v05(xs + 0.5 * T * k1)
    k3 = v05(xs + 0.5 * T * k2)
    k4 = v1(xs + T * k3)

    return xs + T * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def _ode_step_implicit_Euler(
    xs: np.ndarray,
    vector_fields: Tuple[Callable],
    n_iter: int = 1000,
    stopping_rtol: np.float64 = 1e-3,
) -> np.ndarray:

    x_cur = xs
    _, _, v1, T = vector_fields
    for _j in range(n_iter):
        x_new = xs + T * v1(x_cur)
        errs = np.linalg.norm(x_new - x_cur, axis=-1, ord=2)
        xnorms = np.linalg.norm(x_new, axis=-1, ord=2)
        rtols = errs / xnorms
        if np.all(rtols < stopping_rtol):
            break
        x_cur = x_new

    # print(f"{_j=:4d} {rtols.max()=:.2e}")
    return x_new


_name_to_method = {
    "RK45": _ode_step_RK45,
    "EE": _ode_step_explicit_Euler,
    "IE": _ode_step_implicit_Euler,
}

# TODO add density estimation?
def sample_ode(
    xs: np.ndarray,
    vector_fields: List[Tuple[Callable]],
    method: Literal[_name_to_method.keys()] = "RK45",
    *args,
    **kwargs,
) -> np.ndarray:

    ode_step = _name_to_method[method]
    for vf in vector_fields:
        xs = ode_step(xs, vf, *args, *kwargs)

    return xs
