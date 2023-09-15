import numpy as np
import matplotlib.pyplot as plt
import tt

from scipy.stats import norm
from scipy.interpolate import RegularGridInterpolator

from method import *

N = 100
L = 3
dim = 2

grid = np.linspace(-L, L, N)
h_x = grid[1] - grid[0]


rho_0 = tt_independent_gaussians(
    [
        0.0,
    ]
    * dim,
    [
        1.0,
    ]
    * dim,
    grid,
)

n_samples = 1000
xs = norm.rvs(size=(n_samples, dim), random_state=2)
# bc our initial distribution is clipped to cube [-L. L]^d
xs = crop_to_cube(xs, L)

mean_init = xs.mean(axis=0)
cov_init = np.cov(xs, rowvar=False)

m = [1.0, 1.0, 0.0, -0.3][:dim]
sigmas = [
    0.5,
] * dim
rho_infty = tt_independent_gaussians(m, sigmas, grid).full()
# rho_infty = gaussian_mixture([m, -m], [0.5, 0.5], grid)

Ts = [2.0 / _n for _n in range(1, 30)]
betas = [0.5 * _T**1.0 for _T in Ts]

rhos, dKL, dL2, vector_fields = gradient_flow(
    rho_0, rho_infty, grid, Ts, betas, ode_T_max=.1,
)
rho_res = rhos[-1].full()

rho_interp = RegularGridInterpolator(
    (grid,) * dim,
    np.log(rho_res),
    bounds_error=False,
    fill_value=0.0,
)


labels = [
    "explicit Euler",
    "implicit Euler",
    "RK45",
]
methods = [
    "EE",
    "IE",
    "RK45",
]

xx, yy = np.meshgrid(
    *(np.pad(grid, (0, 1), constant_values=(grid[-1] + h_x,)),) * 2,
    indexing="ij",
)

print(f"{mean_init=}")
print(f"{cov_init=}")

for _i in range(3):
    xs_post = sample_ode(xs, vector_fields, method=methods[_i])
    xs_post = crop_to_cube(xs_post, L)

    m_sample = xs_post.mean(axis=0)
    cov_sample = np.cov(
        xs_post.T,
    )

    print(f"{labels[_i]}\n\t{m_sample=}\n\t{cov_sample=}")

    if dim != 2:
        continue

    fig, axs = plt.subplots(1, 1)
    axs.pcolor(xx, yy, rho_res, cmap="Oranges")
    axs.scatter(xs_post[..., 0], xs_post[..., 1], s=0.5)
    axs.set_title(labels[_i])

xx, yy = np.meshgrid(grid, grid, indexing="ij")
coords = np.stack(
    np.meshgrid(
        *(grid,) * dim,
        indexing="ij",
    ),
    axis=-1,
)
m_dens = (coords * rho_res[..., np.newaxis]).sum(axis=tuple(range(dim))) * h_x**dim
coords_centered = coords - m_dens
xixj = coords[..., :, np.newaxis] * coords[..., np.newaxis, :]
cov_dens =(xixj * rho_res[..., np.newaxis, np.newaxis]).sum(axis=tuple(range(dim))) * h_x**dim 

print(f"Density\n\t{m_dens=}\n\t{cov_dens=}")

plt.show()

fig_and_axs = plot_matrix_marginals(
    rho_infty,
    grid,
    color="tab:blue",
    cmap="Blues",
    label=r"$\rho_{\infty}$",
    sym=True,
)
fig_and_axs = plot_matrix_marginals(
    rho_res,
    grid,
    fig_and_axs=fig_and_axs,
    color="tab:orange",
    cmap="Oranges",
    label=r"$\rho_{rWGF}$",
    sample=xs_post,
)

fig, axs = fig_and_axs

plt.show()

# for _i, subscript in enumerate(["0", "\\frac{1}{2} T", "T"]):
#    fig, axs = plt.subplots(1, 1)
#    v = vector_fields[_i]
#    v_at_points = v(coords)
#    v_mag = np.linalg.norm(v_at_points, ord=2, axis=-1, keepdims=True)
#
#    v_dir = v_at_points / v_mag
#
#    _c = axs.quiver(
#        xx,
#        yy,
#        v_dir[..., 0],
#        v_dir[..., 1],
#        v_mag,
#        scale=100.0,
#        width=0.0005,
#        headwidth=10.0,
#        headlength=15,
#        headaxislength=3.0,
#        minlength=1.0,
#        pivot="mid",
#        alpha=0.7,
#    )
#    fig.colorbar(_c, ax=axs)
#    axs.set_title("$v_{" + subscript + "}$")
