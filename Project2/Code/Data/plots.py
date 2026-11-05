import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
from plot_params import COL

FULL = 7.2  # Full page width in inches

'''
Plot helper functions adapted from
https://github.com/EducationalMaterialUiO/MachineLearningUiO/blob/main/doc/WeeklyMaterial/week44/exercisesweek44_solutions.ipynb
'''

def plain_log_ticks(ax, axis="x"):
    '''Log axis with ticks written as 0.001, 0.01, 1, 10 instead of 10^-3 ...'''
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_formatter(fmt)
    a.set_minor_formatter(NullFormatter())


def panel_label(ax, text):
    ax.text(-0.02, 1.04, text, transform=ax.transAxes, fontweight="bold", va="bottom", ha="right")


def noise_floor(ax, sigma=0.1, label=True):
    '''Irreducible MSE on the Runge data = sigma^2'''
    ax.axhline(sigma ** 2, color="gray", ls=":", lw=1.0,
               label=r"noise floor $\sigma^2$" if label else None)
    

def band(ax, a, mean, se, label=None, **kw):
    '''Mean curve with a ±1 SE band'''
    line, = ax.plot(x, mean, label=label, **kw)
    ax.fill_between(x, mean - se, mean + se, color=line.get_color(), alpha=0.2, lw=0)
    return line


def heatmap_marks(ax, within, best=None):
    '''
    Cyan box: minimum. Dotted boxes: cells within one SE of it. NaN = diverged.
    mean, se have shapes (rows, cols, as drawn with imshow/pcolormesh)
    '''
    for r, c in zip(*np.where(within)):
        ax.add_patch(plt.Rectangle((c, r), 1, 1, fill=False, ec="white", ls=":", lw=0.8))
    if best is not None:
        ad.add_patch(plt.Rectangle((best[1], best[0]), 1, 1, fill=False, ec="cyan", lw=1.5))