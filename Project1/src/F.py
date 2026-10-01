import numpy as np
import matplotlib.pyplot as plt
import matplotlib_params
import os
from utils import create_dataset, create_feature_matrix, cost, gradient, closed_form, optimise


filepath = os.path.dirname(os.path.abspath(__file__))
FIGURES_PATH = os.path.join(filepath, '../figures')

ngammas = 60
gammas = np.geomspace(1e-5, 1, ngammas)
np.seterr(over='ignore', invalid='ignore')

max_deg = 9
degrees = np.arange(1, max_deg + 1)
methods = ("plain", "momentum", "adagrad", "rmsprop", "adam")
tol, stride, num_iters = 1e-6, 10, 20000

x, y = create_dataset(n=100, split=False)

scan = {}
for method in methods:
    its = np.full((max_deg, ngammas), np.nan)      # NaN = did not converge
    best = np.full(max_deg, np.nan)                # best gamma per degree
    g_min = np.full(max_deg, np.nan)               # lowest converging gamma
    g_max = np.full(max_deg, np.nan)               # highest converging gamma

    for d in degrees:
        X = create_feature_matrix(x, degree=d)
        gradX = lambda th: gradient(th, X, y)
        cX = cost(closed_form(X, y), X, y)

        for idx, g in enumerate(gammas):
            path, _ = optimise(gradX, np.zeros(d), method, g, num_iters=num_iters)
            e = np.array([np.abs(cost(p, X, y) - cX) for p in path[::stride]])
            e = np.nan_to_num(e, nan=np.inf)       # diverged runs -> not converged
            hit = e < tol
            if hit.any():
                its[d-1, idx] = np.argmax(hit) * stride

        conv = ~np.isnan(its[d-1])
        if conv.any():
            best[d-1] = gammas[np.nanargmin(its[d-1])]   # fewest iterations
            g_min[d-1] = gammas[conv].min()
            g_max[d-1] = gammas[conv].max()

    scan[method] = {"iters": its, "best": best, "min": g_min, "max": g_max}


ncols = 2
nrows = int(np.ceil(len(methods) / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(5.5*ncols, 3.8*nrows),
                         sharex=True, sharey=True, constrained_layout=True)
axes = axes.ravel()

for ax, method in zip(axes, methods):
    s = scan[method]
    ax.fill_between(degrees, s["min"], s["max"], alpha=0.3,
                    label=r"converging $\gamma$ range")
    ax.plot(degrees, s["best"], "o-", label=r"best $\gamma$ (fewest iters)")
    ax.set_yscale("log")
    ax.set_title(method)
    ax.grid(alpha=0.3)

# hide unused panels
for ax in axes[len(methods):]:
    ax.set_visible(False)

# axis labels only on the outer edges
for r in range(nrows):
    axes[r*ncols].set_ylabel(r"initial $\gamma$")
for ax in axes[len(methods)-ncols:len(methods)]:
    ax.tick_params(labelbottom=True)
    ax.set_xlabel("polynomial degree")

axes[0].legend()
plt.savefig(os.path.join(FIGURES_PATH, f'F-Methods_convergence_area.pdf'))

for method in methods:
    s = scan[method]
    print(f"\n=== {method} ===")
    print(f"{'degree':>6} | {'best iters':>10} | {'best gamma':>12}")
    print("-" * 36)
    for d in degrees[:10]:
        row = s["iters"][d-1]
        if np.isnan(row).all():
            print(f"{d:>6} | {'no conv.':>10} | {'-':>12}")
        else:
            print(f"{d:>6} | {int(np.nanmin(row)):>10} | {s['best'][d-1]:>12.3e}")


"""
=== plain ===
degree | best iters |   best gamma
------------------------------------
     1 |         10 |    1.421e-01
     2 |         10 |    2.099e-01
     3 |         70 |    1.727e-01
     4 |        250 |    6.510e-02
     5 |       3990 |    2.454e-02
     6 |      10830 |    9.249e-03
     7 |   no conv. |            -
     8 |   no conv. |            -
     9 |   no conv. |            -

=== momentum ===
degree | best iters |   best gamma
------------------------------------
     1 |         10 |    1.661e-02
     2 |         50 |    1.727e-01
     3 |         70 |    1.366e-02
     4 |         80 |    2.982e-02
     5 |        150 |    5.356e-02
     6 |        590 |    1.661e-02
     7 |      14520 |    6.261e-03
     8 |   no conv. |            -
     9 |   no conv. |            -

=== adagrad ===
degree | best iters |   best gamma
------------------------------------
     1 |         10 |    9.249e-03
     2 |         10 |    3.625e-02
     3 |         10 |    5.356e-02
     4 |        120 |    2.099e-01
     5 |        360 |    6.769e-01
     6 |       8780 |    1.366e-02
     7 |   no conv. |            -
     8 |   no conv. |            -
     9 |   no conv. |            -

=== rmsprop ===
degree | best iters |   best gamma
------------------------------------
     1 |         10 |    8.895e-04
     2 |         10 |    3.486e-03
     3 |         10 |    5.151e-03
     4 |        100 |    2.019e-02
     5 |        230 |    6.510e-02
     6 |       7460 |    7.038e-05
     7 |   no conv. |            -
     8 |   no conv. |            -
     9 |   no conv. |            -

=== adam ===
degree | best iters |   best gamma
------------------------------------
     1 |         10 |    2.360e-03
     2 |         40 |    2.099e-01
     3 |         50 |    4.238e-03
     4 |         70 |    2.019e-02
     5 |        180 |    6.769e-01
     6 |        660 |    7.609e-03
     7 |       5040 |    4.238e-03
     8 |      18520 |    2.360e-03
     9 |   no conv. |            -
"""