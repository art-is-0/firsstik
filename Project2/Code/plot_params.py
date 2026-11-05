import matplotlib.pyplot as plt
from cycler import cycler

COL = 259.48927 / 72.27     # our latex template column width in inches
pad = 4 / 72

plt.rcParams.update({
    # Figure: default is one column wide
    "figure.figsize": (COL, 0.72 * COL),
    "figure.dpi": 500,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.w_pad": pad,
    "figure.constrained_layout.h_pad": pad,
    "savefig.format": "pdf",
    "pdf.fonttype": 42,
    # Fonts
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "figure.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    # Lines and ticks
    "lines.linewidth": 1.2, "lines.markersize": 3.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.3, "ytick.minor.width": 0.3,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.labelpad": 2.0, "axes.titlepad": 4.0,
    # Grid: light grey, behind the data
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "0.85",
    "grid.linewidth": 0.4, "grid.alpha": 1.0,
    # Legend: no frame
    "legend.frameon": False,
    "legend.handlelength": 1.5, "legend.handletextpad": 0.4,
    "legend.columnspacing": 1.0, "legend.borderaxespad": 0.2,
    "patch.linewidth": 0.5,
    # Colors
    "axes.prop_cycle": cycler(color=["steelblue", "palevioletred", "yellowgreen",
                                    "goldenrod", "indianred", "mediumseagreen",
                                    "mediumpurple", "peru"])
})

# plt.rcParams.update(
#     {
#         # Figure
#         "figure.dpi": 600,
#         "figure.constrained_layout.use": True,
#         "figure.constrained_layout.w_pad": pad,
#         "figure.constrained_layout.h_pad": pad,
#         "savefig.format": "pdf",
#         # Plotting
#         "lines.linewidth": 2,
#         # Axes
#         "axes.linewidth": 0.5,
#         #"axes.grid": True,
#         "grid.color": "black",
#         "grid.alpha": 0.25,
#         "axes.labelpad": 2.0,
#         "axes.titlepad": 5.0,
#         "ytick.major.width": 0.6,
#         "xtick.major.width": 0.6,
#         "ytick.minor.width": 0.3,
#         "xtick.minor.width": 0.3,
#         # Fonts
#         "legend.fontsize": font_size,
#         "axes.titlesize": font_size,
#         "figure.titlesize": font_size + 1,
#         "axes.labelsize": font_size,
#         "xtick.labelsize": font_size - 2,
#         "ytick.labelsize": font_size - 2,
#         # Legend
#         "legend.handletextpad": 0.3,
#         "legend.scatterpoints": 3,
#         "legend.borderaxespad": 0.1,
#         "legend.fancybox": False,
#         "patch.linewidth": 0.5,
#         "legend.columnspacing": 1,
#         # "legend.edgecolor": "white",
#     }
# )