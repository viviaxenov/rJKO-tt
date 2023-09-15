import numpy as np
from scipy.stats import ortho_group

import tt

from method import *
from utility import *

np.seterr(all="raise")

N = 100
dim = 3
L = 3
print(np.sqrt(N ** (dim - 1) / dim))

grid = np.linspace(-L, L, N)
h_x = grid[1] - grid[0]

rho_0 = tt_independent_gaussians(
    [0.0] * dim,
    [1.0] * dim,
    grid,
)

N_comp = 5
means = [
    np.array(
        [
            1.5 * np.cos(2.0 * np.pi / N_comp * _i),
            1.5 * np.sin(2.0 * np.pi / N_comp * _i),
            -1.0 + 2.0 / N_comp * _i,
        ]
    )
    for _i in range(N_comp)
]
sigmas = [
    0.25,
] * N_comp


rho_inf = gaussian_mixture(means, sigmas, grid)
rho_inf_tt = tt.vector(rho_inf, eps=tt_precision)

print(rho_inf_tt.r)

# Ts = [
#     1.0,
#     1.0,
#     1.0,
#     1.0,
#     0.5,
#     0.5,
#     0.5,
#     0.5,
#     0.1,
#     0.1,
#     0.1,
#     0.1,
#     0.05,
#     0.05,
#     0.05,
#     0.05,
# ]  # 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, ]
# betas = [interp_dict.get(_T, 0.03) for _T in Ts]

Ts = [2.0 / _n for _n in range(1, 3)]
betas = [0.5 * _T**1.0 for _T in Ts]

# betas = [0.05 for _T in Ts]
# betas = [0.005*(_T/0.05)**0.5 for _T in Ts]
# betas = [interp(_T) for _T in Ts]

# Ts = Ts[:1]
# betas = betas[:1]


# plt.plot(Ts, betas)
# plt.show()
# exit()

rhos, dKL, dL2 = gradient_flow(rho_0, rho_inf, grid, Ts, betas, 3.5e-3, )

iterations_made = len(dKL) - 1

Ts = Ts[:iterations_made]
betas = betas[:iterations_made]

rel_err_L2 = dL2[-1] / tt.vector.norm(rhos[-1])

print(f"{dKL[-1]=:.2e}, {dL2[-1]=:.2e} {rel_err_L2=:.2e}")

fig_path = "./3d_correlated_res/"

plots_gradient_flow(
    rhos,
    rho_0,
    rho_inf,
    grid,
    Ts,
    dKL,
    dL2,
    fig_path=fig_path,
)
fig_and_axs = plot_matrix_marginals(
    rho_inf,
    grid,
    color="tab:blue",
    cmap="Blues",
    label=r"$\rho_{\infty}$",
)
fig_and_axs = plot_matrix_marginals(
    rhos[-1],
    grid,
    fig_and_axs=fig_and_axs,
    color="tab:orange",
    cmap="Oranges",
    label=r"$\rho_{rWGF}$",
)

fig, axs = fig_and_axs

plt.show()
