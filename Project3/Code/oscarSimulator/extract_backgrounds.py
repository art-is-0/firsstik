"""Extract SNIP backgrounds from the per-detector LaBr3 spectra in an OSCAR .root file.

Reads a TH2 (energy x detector) histogram, smooths each detector's spectrum
(Gaussian, --smooth-sigma), runs snipBackground (NumPy) with a half-window that
grows with the LaBr3 FWHM (--fwhm-factor x FWHM(E)), and saves spectra and
backgrounds to a .npz file. The defaults (sigma 5 keV, 1.5 x FWHM) were chosen by
eye on detectors 0 and 9 of the sirius-20190313 file; check the plots for other runs.
The .root file is read with uproot, so ROOT itself is not needed (see requirements.txt).

Optional check, needs a ROOT installation with PyROOT: --compare-root also runs ROOT's
TSpectrum::Background with a fixed window on the same input and prints the largest
difference from the NumPy code. ROOT's thisroot.sh does not work in every shell; if it
fails, set the variables directly:
    export ROOTSYS=$HOME/root PATH=$HOME/root/bin:$PATH \
           LD_LIBRARY_PATH=$HOME/root/lib:$LD_LIBRARY_PATH PYTHONPATH=$HOME/root/lib:$PYTHONPATH

Run:
    ~/machinelearning/bin/python extract_backgrounds.py data/measured/si28-sirius-20190313-142346.root \
        --plot-detector 0

Output .npz keys:
    keVBinCentres (channels,), counts (detectors, channels),
    background (detectors, channels), detectorIndex (detectors,), plus the settings.

LLM usage: Level 4 (Substantial); an LLM wrote this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import uproot
from scipy.ndimage import gaussian_filter1d

from background_estimation import snipBackground
from data_reader import BACKGROUNDS_DIR, FIG_DIR


def readSpectraPerDetector(rootPath, histogramName):
    """Read a TH2 (x: energy or channel, y: detector) into NumPy arrays.

    returns:
        binCentres: (channels,plot) x-axis bin centres (keV for the calibrated histogram)
        counts: (detectors, channels) counts per bin for each detector
        axisTitle: x-axis title, to record the unit

    LLM usage: Level 4 (Substantial).
    """
    with uproot.open(rootPath) as rootFile:
        if histogramName not in rootFile:
            available = ", ".join(sorted({key.split(";")[0] for key in rootFile.keys()}))
            raise KeyError(f"No histogram '{histogramName}' in {rootPath}. Available: {available}")
        histogram = rootFile[histogramName]
        counts = np.asarray(histogram.values(), dtype=float).T  # values() is (x, y) without flow bins
        xAxis = histogram.axis(0)
        binCentres = np.asarray(xAxis.centers(), dtype=float)
        axisTitle = str(xAxis.member("fTitle"))
    return binCentres, counts, axisTitle


def rootSnipBackground(counts, maxHalfWindow):
    """ROOT's TSpectrum::Background with the settings snipBackground reproduces.

    Only used by --compare-root; needs PyROOT, which is not in requirements.txt.

    LLM usage: Level 4 (Substantial).
    """
    import ROOT  # imported here so the script runs without ROOT unless this check is asked for

    spectrum = np.ascontiguousarray(counts, dtype=np.float64).copy()
    ROOT.TSpectrum().Background(
        spectrum, spectrum.size, maxHalfWindow,
        ROOT.TSpectrum.kBackIncreasingWindow, ROOT.TSpectrum.kBackOrder2,
        False, ROOT.TSpectrum.kBackSmoothing3, False,
    )
    return spectrum


def plotDetector(binCentres, counts, background, detector, axisTitle, outputPath):
    """Save a log-scale plot of one detector's spectrum with its SNIP background.

    LLM usage: Level 4 (Substantial).
    """
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(9, 4.5))
    axes.step(binCentres, counts, where="mid", lw=0.6, label=f"detector {detector}")
    axes.plot(binCentres, background, lw=1.2, label="SNIP background")
    axes.set_yscale("log")
    axes.set_ylim(bottom=0.5)
    axes.set_xlabel(axisTitle or "channel")
    axes.set_ylabel("counts per bin")
    axes.legend()
    figure.tight_layout()
    figure.savefig(outputPath, dpi=150)
    plt.close(figure)


def main() -> None:
    """Extract backgrounds for all detectors and save them.

    LLM usage: Level 4 (Substantial).
    """
    argumentParser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    argumentParser.add_argument("root_path", type=Path)
    argumentParser.add_argument("--histogram", default="energy_cal_labr",
                                help="TH2 with energy on x and detector on y "
                                     "(energy_cal_labr: 1 keV bins; energy_labr: raw channels)")
    argumentParser.add_argument("--fwhm-factor", type=float, default=1.5,
                                help="per-channel SNIP half-window = factor x FWHM(E) / bin width "
                                     "(needs a keV axis; uses LaBr3Detector.fwhm)")
    argumentParser.add_argument("--half-window", type=int, default=None,
                                help="one fixed SNIP half-window in bins instead of --fwhm-factor "
                                     "(required for the raw-channel histogram)")
    argumentParser.add_argument("--smooth-sigma", type=float, default=5.0,
                                help="Gaussian smoothing (in bins) before clipping, against the "
                                     "low bias from Poisson fluctuations; 0 turns it off")
    argumentParser.add_argument("--lls", action="store_true",
                                help="use the log-log-sqrt transform before clipping")
    argumentParser.add_argument("--compare-root", action="store_true",
                                help="also run ROOT's TSpectrum::Background and print the largest "
                                     "difference (optional check; needs PyROOT)")
    argumentParser.add_argument("--plot-detector", type=int, default=None,
                                help="save a plot of this detector's spectrum and background")
    argumentParser.add_argument("--output", type=Path, default=None,
                                help="output .npz (default: data/backgrounds/backgrounds_<root file stem>.npz)")
    args = argumentParser.parse_args()

    binCentres, counts, axisTitle = readSpectraPerDetector(args.root_path, args.histogram)
    print(f"Read '{args.histogram}': {counts.shape[0]} detectors x {counts.shape[1]} bins "
          f"(x axis: {axisTitle!r})")

    if args.half_window is not None:
        halfWindow = np.full(binCentres.size, args.half_window, dtype=int)
    else:
        from detector import LaBr3Detector

        binWidth = binCentres[1] - binCentres[0]
        keVFWHM = LaBr3Detector(None).fwhm(binCentres)
        halfWindow = np.maximum(np.rint(args.fwhm_factor * keVFWHM / binWidth).astype(int), 3)
        print(f"Half-window {args.fwhm_factor} x FWHM(E): {halfWindow[0]} to {halfWindow[-1]} bins")

    smoothedCounts = (gaussian_filter1d(counts, args.smooth_sigma, axis=1)
                      if args.smooth_sigma > 0 else counts)
    background = np.array([
        snipBackground(spectrum, halfWindow, useLogLogSqrt=args.lls) for spectrum in smoothedCounts
    ])

    if args.compare_root:
        # TSpectrum has one fixed window and no LLS, so compare the shared core algorithm
        fixedWindow = args.half_window or 40
        largestDifference = max(
            np.max(np.abs(snipBackground(spectrum, fixedWindow)
                          - rootSnipBackground(spectrum, fixedWindow)))
            for spectrum in smoothedCounts
        )
        print(f"Check, fixed window {fixedWindow}: largest |NumPy SNIP - TSpectrum::Background| "
              f"over all detectors = {largestDifference:.3g} counts")

    hasCounts = counts.sum(axis=1) > 0
    for detector in np.flatnonzero(~hasCounts):
        print(f"Detector {detector} is empty")

    outputPath = args.output or BACKGROUNDS_DIR / f"backgrounds_{args.root_path.stem}.npz"
    np.savez_compressed(
        outputPath,
        keVBinCentres=binCentres, counts=counts, background=background,
        detectorIndex=np.arange(counts.shape[0]),
        sourceFile=str(args.root_path), histogram=args.histogram, axisTitle=axisTitle,
        maxHalfWindow=halfWindow, fwhmFactor=args.fwhm_factor if args.half_window is None else 0.0,
        smoothSigma=args.smooth_sigma, useLogLogSqrt=args.lls,
    )
    print(f"Saved to {outputPath}")

    if args.plot_detector is not None:
        plotPath = FIG_DIR / f"{outputPath.stem}_detector{args.plot_detector}.png"
        plotDetector(binCentres, counts[args.plot_detector], background[args.plot_detector],
                     args.plot_detector, axisTitle, plotPath)
        print(f"Plot saved to {plotPath}")


if __name__ == "__main__":
    main()
