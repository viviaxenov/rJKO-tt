import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import expm
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt
import matplotlib.animation as ani
import matplotlib.colors as colors

import tt
import tt.amen
import tt.multifuncrs
import tt.cross.rectcross

from functools import cache
from time import perf_counter
import traceback

import os.path

from typing import Literal, Callable, Tuple, List, Union

from itertools import cycle


np.seterr(all="raise")




def crop_to_cube(xs: np.ndarray, L: np.float64) -> np.ndarray:
    select_indices = np.all(xs <= L, axis=-1) & np.all(xs >= -L, axis=-1)
    xs = xs[select_indices, :]

    return xs


def tt_sum_multi_axis(a, axis=-1):
    """
    Sum TT-vector over specified axes
    Condtion marked by * below were misplaced in the original code;
    should I make pull request?
    """
    d = a.d
    crs = tt.vector.to_list(a)
    if isinstance(axis, int):
        if axis < 0:  # (*)
            axis = range(a.d)
        else:
            axis = [axis]
    axis = list(axis)[::-1]
    for ax in axis:
        crs[ax] = np.sum(crs[ax], axis=1)
        rleft, rright = crs[ax].shape
        if (rleft >= rright or rleft < rright and ax + 1 >= d) and ax > 0:
            crs[ax - 1] = np.tensordot(crs[ax - 1], crs[ax], axes=(2, 0))
        elif ax + 1 < d:
            crs[ax + 1] = np.tensordot(crs[ax], crs[ax + 1], axes=(1, 0))
        else:
            return np.sum(crs[ax])
        crs.pop(ax)
        d -= 1
    return tt.vector.from_list(crs)


def tt_take(a: tt.vector, indices: np.ndarray):
    """
    Slicing with a list of indices; needed for cross approximation algorithm

    Args:
        a (tt.vector): tensor-train with dimension D
        indices (np.ndarray): array with shape [I, D] where I is the number of values to be taken
    Returns:
        np.ndarray: array of shape [I,] with values of a at said indices
    """
    return np.array([a[_idx] for _idx in indices])


def get_grid_function(f: Callable, grid1d: np.ndarray):
    """
    Returns a restriction of f on a prescribed grid [grid1d]^d

    Args:
        f (Callable): function R^d -> R
        grid1d (np.ndarray): unidimensional grid

    Returns:
        Callable: function that takes [I, d] array of integer indices and returns [I] array of [f(x_1^i, ... x_d^i))]
    """

    def _f_on_grid(indices: np.ndarray):

        return f(grid1d[indices])

    return _f_on_grid


def tt_hadamard_product(tensor1: tt.vector, tensor2: tt.vector):
    t1_as_oper = tt.matrix.from_list(
        [
            np.einsum("ij,kil->kijl", np.eye(_v.shape[1]), _v)
            for _v in tt.vector.to_list(tensor1)
        ]
    )

    res = tt.matvec(
        t1_as_oper,
        tensor2,
    )
    return res.round(
        eps=1e-12,
    )


def tt_full_like(a_tt: tt.vector, fill_value: np.float64):
    return tt.vector.from_list(
        [
            np.full((1, _v.shape[1], 1), fill_value=fill_value)
            for _v in tt.vector.to_list(a_tt)
        ]
    )


def div_KL(
    rho1: tt.vector, rho2: Union[np.ndarray, Callable], h_x: np.float64
) -> np.float64:
    dim = rho1.d
    if isinstance(rho2, np.ndarray):

        if isinstance(rho1, tt.vector):
            rho1 = rho1.full()

        eps = 1.0 - np.sum(rho1) * h_x**dim
        try:
            V = (
                np.sum(
                    (np.log(np.where(rho1 > 1e-15, rho1, 1e-15)) - np.log(rho2)) * rho1
                )
                * h_x**dim
            )
            return V + eps
        except:
            return -1.0

    else:
        print("Estimating KL err", flush=True)
        log_quot = tt.cross.rectcross.cross(
            lambda _S: np.log(np.maximum(tt_take(rho1, _S) / rho2(_S), 1e-15)),
            tt_full_like(rho1, 0.0),
            eps=1e-4,
            nswp=5,
        )
        print("", flush=True)
        return tt.dot(log_quot, rho1) * h_x**dim


def KL_stupid(
    eta: tt.vector, hat_eta: tt.vector, beta: np.float64, h_x: np.float64
) -> np.float64:
    dim = eta.d
    log_quot = tt.multifuncrs2(
        [eta],
        lambda _x: np.log(np.maximum(1e-16, _x)),
        eps=1e-5,
        kickrank=2,
        nswp=8,
    )
    rho = tt_hadamard_product(eta, hat_eta)

    return -2.0 * beta * tt.dot(log_quot, rho) * h_x**dim


def div_L2(
    rho1: tt.vector, rho2: Union[np.ndarray, Callable], h_x: np.float64
) -> np.float64:
    dim = rho1.d
    if isinstance(rho2, np.ndarray):
        rho1 = rho1.full()
        return np.linalg.norm((rho1 - rho2).ravel(), ord=2)
    else:
        print("Estimating L2 err", flush=True)
        diff = tt.cross.rectcross.cross(
            lambda _S: (tt_take(rho1, _S) - rho2(_S)),
            tt_full_like(rho1, 0.0),
            eps=1e-4,
            kickrank=2,
            nswp=5,
        )
        return np.sqrt(tt.dot(diff, diff))


gauss_density_fn = lambda _x, _m, _sigma: np.exp(-0.5 * ((_x - _m) / _sigma) ** 2) / (
    np.sqrt(2.0 * np.pi) * _sigma
)


def tt_independent_gaussians(
    ms: List[np.float64], sigmas: List[np.float64], grid: np.ndarray
) -> tt.vector:

    assert len(ms) == len(sigmas)

    d = len(ms)
    h_x = grid[1] - grid[0]
    N = grid.shape[0]
    nodes = [
        gauss_density_fn(grid, ms[i], sigmas[i]).reshape(1, N, 1) for i in range(d)
    ]
    rho = tt.vector.from_list(nodes)
    rho *= 1.0 / (tt.sum(rho) * h_x**d)
    return rho


def get_correlated_gaussian_fn(
    mean: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:

    dim = mean.shape[0]
    assert dim == covariance.shape[0] and dim == covariance.shape[1]

    precision = np.linalg.inv(covariance)
    scale = (2.0 * np.pi) ** (dim / 2.0) * np.linalg.det(covariance) ** 0.5

    mean = mean.reshape((1, -1))

    def _dens(x: np.ndarray):  # shape[0] = dim,
        x_centered = x - mean
        rho = (
            np.exp(
                -0.5 * np.einsum("...i,ij,...j->...", x_centered, precision, x_centered)
            )
            / scale
        )

        return rho

    return _dens


def get_gaussian_mixture_fn(
    means: List[np.ndarray],
    covariances: List[Union[np.ndarray, np.float64]],
    weights: np.ndarray = None,
):
    n_comp = len(means)
    assert len(covariances) == n_comp
    if weights is not None:
        assert len(weights) == n_comp
        assert np.all(weights > 0)
        weights /= np.sum(weights)
    else:
        weights = np.full((n_comp,), 1.0 / n_comp)

    dim = means[0].shape[0]

    comps = []

    for _i, mean in enumerate(means):
        assert mean.shape[0] == dim
        cov = covariances[_i]
        if isinstance(cov, float):
            cov = np.eye(dim) * cov
        else:
            assert cov.shape[0] == dim
            if len(cov.shape) != 2:
                cov = np.diag(cov)
            else:
                assert cov.shape[1] == dim

        comps.append(get_correlated_gaussian_fn(mean, cov))

    def _density(x: np.ndarray):
        comp_vals = np.stack([_comp(x) for _comp in comps])
        return np.einsum("i,i...", weights, comp_vals)

    return _density


def plot_1d_marginals(
    density: Union[tt.vector, np.ndarray],
    grid: np.ndarray,
    fig_and_axs=None,
    *plot_args,
    **plot_kwargs,
):

    dim = density.d if isinstance(density, tt.vector) else len(density.shape)
    h_x = grid[1] - grid[0]

    if fig_and_axs in [None, (None, None)]:
        fig, axs = plt.subplots(1, dim, sharey=True)
        fig.suptitle("Marginals")
    else:
        fig, axs = fig_and_axs

    for n_marginal in range(dim):
        other_axes = set(range(dim))
        other_axes.remove(n_marginal)
        other_axes = tuple(other_axes)

        density_marginal = (
            tt_sum_multi_axis(density, axis=other_axes).full() * h_x ** (dim - 1)
            if isinstance(density, tt.vector)
            else np.sum(density, axis=other_axes) * h_x ** (dim - 1)
        )

        axs[n_marginal].plot(grid, density_marginal, *plot_args, **plot_kwargs)
        axs[n_marginal].grid(True)
        axs[n_marginal].set_xlabel(f"$x_{n_marginal+1}$")

    axs[-1].legend()
    return (fig, axs)


def plot_2d_marginals(
    density: Union[tt.vector, np.ndarray],
    grid: np.ndarray,
    marginals=(0, 1),
    fill=False,
    fig_and_axs=None,
    *plot_args,
    **plot_kwargs,
):

    dim = density.d if isinstance(density, tt.vector) else len(density.shape)
    h_x = grid[1] - grid[0]

    fig: plt.Figure
    axs: plt.Axes

    if fig_and_axs in [None, (None, None)]:
        fig, axs = plt.subplots(1, 1)
        fig.suptitle(
            f"Joint distribuition of $x_{marginals[0]}$ and $x_{marginals[1]}$"
        )
        axs.set_xlabel(f"$x_{marginals[0]}$")
        axs.set_ylabel(f"$x_{marginals[1]}$")

    else:
        fig, axs = fig_and_axs

    other_axes = set(range(dim))
    other_axes.difference_update(marginals)
    other_axes = tuple(other_axes)

    density_marginal = (
        tt_sum_multi_axis(density, axis=other_axes).full()
        if isinstance(density, tt.vector)
        else np.sum(density, axis=other_axes)
    ) * h_x ** (dim - 2)

    if marginals[0] > marginals[1]:
        density_marginal = density_marginal.T

    plot_fn = axs.contourf if fill else axs.contour

    contours = plot_fn(
        *np.meshgrid(grid, grid, indexing="ij"),
        density_marginal,
        10,
        *plot_args,
        **plot_kwargs,
    )
    return fig, axs


def plot_matrix_marginals(
    density: Union[tt.vector, np.ndarray],
    grid: np.ndarray,
    sample=None,
    fig_and_axs=None,
    cmap="Blues",
    sym=False,
    *plot_args,
    **plot_kwargs,
):
    dim = density.d if isinstance(density, tt.vector) else len(density.shape)
    h_x = grid[1] - grid[0]

    # fig, axs = plt.subplots(dim, dim, sharex='row', sharey='col')
    if fig_and_axs in [None, (None, None)]:
        fig, axs = plt.subplots(
            dim,
            dim,
        )
    else:
        fig, axs = fig_and_axs

    for _i in range(dim):
        axs[_i, 0].set_ylabel(f"$x_{_i + 1}$")
        for _j in range(_i + 1, dim):
            plot_2d_marginals(
                density,
                grid,
                fig_and_axs=(fig, axs[_i, _j]),
                marginals=(_i, _j),
                cmap=cmap,
            )
            if sym:
                plot_2d_marginals(
                    density,
                    grid,
                    fig_and_axs=(fig, axs[_j, _i]),
                    marginals=(_j, _i),
                    cmap=cmap,
                    fill=True,
                )

    for _j in range(dim):
        axs[-1, _j].set_xlabel(f"$x_{_j + 1}$")

    color = plot_kwargs["color"]
    if sample is not None:
        for _i in range(dim):
            for _j in range(_i + 1, dim):
                axs[_j, _i].scatter(sample[:, _j], sample[:, _i], c=color, s=0.5)

    axs_diag = np.diag(axs)
    plot_1d_marginals(density, grid, (fig, axs_diag), *plot_args, **plot_kwargs)
    for _i in range(dim):
        axs[_i, _i].set_ylabel(f"$\\rho_{_i+1}$")
    for _i in range(1, dim):
        axs[_i, _i].sharey(axs[_i - 1, _i - 1])

    return fig, axs


# TODO: encapsulation
def plots_gradient_flow(
    rhos: List[tt.vector],
    rho_0: tt.vector,
    rho_inf: np.ndarray,
    grid: np.ndarray,
    Ts: List[np.float64],
    dKL: List[np.float64],
    dL2: List[np.float64],
    fig_path=None,
):

    dim = rho_0.d
    h_x = grid[1] - grid[0]
    time = np.cumsum(np.array([0.0] + Ts))

    fig, axs = plt.subplots(1, 1)

    axs.plot(time, dKL, label="KL", marker="*")
    axs.plot(time, dL2, label="L2", marker="^")
    axs.set_yscale("log")
    axs.set_xlabel("Time")
    axs.set_title("Convergence")
    axs.legend()
    axs.grid()

    if fig_path is not None:
        fig.savefig(
            os.path.join(
                fig_path,
                "gf_covnergence.pdf",
            )
        )
    else:
        plt.show()

    fig_and_axs = None
    fig_and_axs = plot_1d_marginals(rho_0, grid, fig_and_axs, label="Initial")
    fig_and_axs = plot_1d_marginals(rho_inf, grid, fig_and_axs, label="Reference")
    fig_and_axs = plot_1d_marginals(rhos[-1], grid, fig_and_axs, "g*", label="Result")

    fig, axs = fig_and_axs

    if fig_path is not None:
        fig.savefig(
            os.path.join(
                fig_path,
                "gf_marginals.pdf",
            )
        )
    else:
        plt.show()

    fig_and_axs = None
    marginals = (0, 1)
    fig_and_axs = plot_2d_marginals(
        rho_inf,
        grid,
        fig_and_axs=fig_and_axs,
        label="Reference",
        cmap="Blues",
    )
    fig_and_axs = plot_2d_marginals(
        rhos[-1],
        grid,
        fig_and_axs=fig_and_axs,
        label="Result",
        cmap="Oranges",
        vmin=1e-5,
    )
    fig, axs = fig_and_axs
    axs.legend()
    fig.tight_layout()

    if fig_path is not None:
        fig.savefig(
            os.path.join(
                fig_path,
                f"gf_2d_marginals{marginals[0]}{marginals[1]}.pdf",
            )
        )
    else:
        plt.show()

    return


def plot_convergence(
    Ts: List[np.float64],
    err_stats: dict,
    fig_path=None,
):

    time = np.cumsum(np.array([0.0] + Ts))
    fig, axs = plt.subplots(1, 1)
    markers = cycle(["*", "^", "s", "o"])

    for _stat in err_stats:
        axs.plot(time, err_stats[_stat], label=_stat, marker=next(markers))
    axs.set_yscale("log")
    axs.set_xlabel("Time")
    axs.set_title("Convergence")
    axs.legend()
    axs.grid()

    if fig_path is not None:
        fig.savefig(
            os.path.join(
                fig_path,
                "gf_covnergence.pdf",
            )
        )
    else:
        plt.show()

    return


def plots_one_step(
    grid: np.ndarray,
    eta: tt.vector,
    hat_eta: tt.vector,
    abs_errs: List[np.float64],
    rel_errs: List[np.float64],
    relaxations: List[np.float64],
    ranks: List[int],
    fig_path=None,
    plot_vector_field=True,
):
    h_x = grid[1] - grid[0]
    dim = eta.d
    assert dim == 2, "Sorry, visualization only works in 2-d"

    rho_1 = tt_hadamard_product(eta, hat_eta)
    fig, axs = plt.subplots(nrows=1, ncols=4)

    _c = axs[0].pcolor(rho_1.full())
    fig.colorbar(_c, ax=axs[0])

    axs[0].set_title("$\\rho(T=1)$")

    # _c = axs[1].pcolor(
    #     eta.full(),
    #     norm=colors.LogNorm(),
    # )
    _c = axs[1].scatter(
        xx,
        yy,
        c=eta.full(),
        marker="s",
        s=25.0,  # norm=colors.LogNorm()
    )
    fig.colorbar(_c, ax=axs[1])
    axs[1].set_title("$\\eta(T=1)$")

    # _c = axs[2].pcolor(
    #     hat_eta.full(),
    #     # norm=colors.LogNorm(),
    # )

    _c = axs[2].scatter(
        xx,
        yy,
        c=hat_eta.full(),
        marker="s",
        s=25.0,
        # norm=colors.SymLogNorm(linthresh=1e-5),
    )
    fig.colorbar(_c, ax=axs[2])

    axs[2].set_title("$\\hat\\eta(T=1)$")

    labels = ["Absolute", "Relative"]

    for idx, data in enumerate([abs_errs, rel_errs]):
        axs[-1].plot(data, label=labels[idx])

    axs[-1].set_yscale("log")
    axs[-1].grid()
    axs[-1].legend()

    ax_rk = axs[-1].twinx()
    ax_rk.plot(ranks, "r-", label="max TT-rank")
    ax_rk.grid()
    ax_rk.legend()

    if fig_path is not None:
        fig.savefig(
            os.path.join(
                fig_path,
                "variables.pdf",
            )
        )
    else:
        mplcursors.cursor(hover=True)
        plt.show()

    dens = rho_1.full() * h_x**dim

    eta_dense = eta.full()
    eta_dense = np.where(eta_dense > 1e-15, eta_dense, 1e-15)

    velocity_potential = np.pad(
        2.0 * beta * (np.log(eta_dense) - np.log(hat_eta.full())),
        pad_width=1,
        mode="edge",
    )

    v_x = (velocity_potential[2:, 1:-1] - velocity_potential[:-2, 1:-1]) / h_x
    v_y = (velocity_potential[1:-1, 2:] - velocity_potential[1:-1, :-2]) / h_x

    velocity = np.stack((v_x, v_y))
    velocity_mag = np.sqrt(v_x**2 + v_y**2)
    velocity_dir = velocity / velocity_mag

    fig, axs = plt.subplots(1, 1)

    _c = axs.pcolor(velocity_mag)
    axs.quiver(
        # xx, # anchor coords for arrows (different for cells in pcolormesh so do later)
        # yy,
        *velocity_dir,
        # velocity_mag, coloring
        scale=100.0,
        width=0.0005,
        headwidth=10.0,
        headlength=15,
        headaxislength=3.0,
        minlength=1.0,
        pivot="mid",
        alpha=0.7,
    )
    fig.colorbar(_c, ax=axs)

    if fig_path is not None:
        fig.savefig(os.path.join(fig_path, "vector_field.pdf"))
    else:
        plt.show()
