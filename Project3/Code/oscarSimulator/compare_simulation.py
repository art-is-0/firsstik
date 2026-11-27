"""Plot a simulated spectrum next to a measured OSCAR spectrum, for a visual check of the simulator.

The simulated spectrum uses known full-energy peak energies for the target of the measured
dataset (read from the file name: si28-..., 27Al_..., 106Pd_...), plus contaminants and the
511 keV and La-138 peaks. It is built in the same binning as the calibrated measured spectrum
(1 keV for energy_cal_labr, 2 keV for h_LaBr3_energy), so the two can be overlaid in keV.

No background is extracted from the dataset itself: the background SHAPE is always the one
fitted to the 28Si run (data/backgrounds, fitBackgroundCoefficients). Peak areas and the
background level are found together by one weighted non-negative least-squares fit of
templates to the measured counts:
    - one template per peak: Gaussian full-energy peak + Compton continuum + escape peaks
      (simulator widths and ratios, all random factors 1)
    - one template for the background shape (fitted polynomial above 500 keV, the measured
      28Si low-energy shape with its hump below, at its measured strength)
    - all templates multiplied by the threshold and upper cut-off
The simulated spectrum is this fitted model plus Poisson noise, so the plot shows how well the
simulator's physics (widths, Compton continua, escape peaks) and the 28Si background shape
describe each dataset.

Run (from proj3/):
    ~/machinelearning/bin/python compare_simulation.py                          # 28Si, detector 0
    ~/machinelearning/bin/python compare_simulation.py 27Al_cal2.root --detector 3
    ~/machinelearning/bin/python compare_simulation.py all                      # every file in data/measured
    ~/machinelearning/bin/python compare_simulation.py 106Pd_27th.root --peaks 511.9 616.2 717.4

Figures are saved to figures/compare_simulation_<target>_detector<N>.png.

LLM usage: Level 4 (Substantial); an LLM wrote this script.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uproot
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import nnls

import simulate as sim
from data_reader import FIG_DIR, MEASURED_DIR

# Gamma-ray energies (keV). Check them against ENSDF and against the measured spectra before
# drawing conclusions; add or remove peaks with --peaks.
SI28_PEAK_ENERGIES_KEV = [
    1779.0,   # 2+ 1779 -> 0+ ground state
    2838.3,   # 4+ 4618 -> 2+ 1779
    3200.7,   # 0+ 4980 -> 2+ 1779
    4497.4,   # 6276 -> 1779
    5099.7,   # 3- 6879 -> 1779
    5600.6,   # 2+ 7381 -> 1779
    6878.8,   # 3- 6879 -> 0+
    7380.6,   # 2+ 7381 -> 0+
]

# 27Al peaks (a contaminant in the 28Si run)
AL27_PEAK_ENERGIES_KEV = [
    843.8,    # 1/2+ 844 -> 5/2+ ground state
    1014.4,   # 3/2+ 1014 -> 5/2+ ground state
    2211.0,   # 7/2+ 2211 -> 5/2+ ground state
]

# Further 27Al transitions, used when 27Al is the target
AL27_TARGET_EXTRA_PEAK_ENERGIES_KEV = [
    1720.3,   # 2735 -> 1014
    2734.9,   # 5/2+ 2735 -> ground state
    3004.2,   # 9/2+ 3004 -> ground state
]

# Reaction products seen in the 27Al run
MG24_PEAK_ENERGIES_KEV = [1368.7]   # 2+ 1369 -> 0+ in 24Mg
MG26_PEAK_ENERGIES_KEV = [1808.7]   # 2+ 1809 -> 0+ in 26Mg

PD106_PEAK_ENERGIES_KEV = [
    511.9,    # 2+ 512 -> 0+ (overlaps the 511 keV annihilation peak)
    616.2,    # 2+ 1128 -> 2+ 512
    621.9,    # 0+ 1134 -> 2+ 512
    717.4,    # 4+ 1229 -> 2+ 512
    1050.4,   # 2+ 1562 -> 2+ 512
    1128.0,   # 2+ 1128 -> 0+
]

# Contaminants: oxygen in the target or its oxide layer, carbon build-up on the target
O16_PEAK_ENERGIES_KEV = [
    6129.9,   # 3- 6130 -> 0+ ground state
    6917.1,   # 2+ 6917 -> 0+ ground state
    7115.2,   # 1- 7117 -> 0+ ground state
]
C12_PEAK_ENERGIES_KEV = [
    4438.9,   # 2+ 4439 -> 0+ ground state (often Doppler-broadened in-beam)
]
CONTAMINANT_PEAK_ENERGIES_KEV = O16_PEAK_ENERGIES_KEV + C12_PEAK_ENERGIES_KEV

# Per target: plot label, default peaks, where the measured spectrum ends, zoom range of panel 2
TARGETS = {
    "Si28": {
        "label": r"$^{28}$Si",
        "peaks": SI28_PEAK_ENERGIES_KEV + AL27_PEAK_ENERGIES_KEV + CONTAMINANT_PEAK_ENERGIES_KEV,
        "keVUpperCutoff": 11300.0,
        "keVZoom": (300.0, 3500.0),
    },
    "Al27": {
        "label": r"$^{27}$Al",
        "peaks": (AL27_PEAK_ENERGIES_KEV + AL27_TARGET_EXTRA_PEAK_ENERGIES_KEV + MG24_PEAK_ENERGIES_KEV
                  + MG26_PEAK_ENERGIES_KEV + CONTAMINANT_PEAK_ENERGIES_KEV),
        "keVUpperCutoff": 9900.0,
        "keVZoom": (300.0, 3500.0),
    },
    "Pd106": {
        "label": r"$^{106}$Pd",
        "peaks": PD106_PEAK_ENERGIES_KEV + CONTAMINANT_PEAK_ENERGIES_KEV,
        "keVUpperCutoff": 5200.0,  # the 106Pd spectrum has hardly any counts above about 5 MeV
        "keVZoom": (300.0, 2000.0),
    },
}

DEFAULT_DATASET = "si28-sirius-20190313-142346.root"
HISTOGRAM_NAMES = ["energy_cal_labr", "h_LaBr3_energy"]  # first one found in the file is used


def targetFromFilename(path):
    """Target key such as "Si28" from a file name like si28-sirius-..., 27Al_cal2 or 106Pd_27th.

    Accepts both orders (mass number first or element first), any capitalisation.
    Returns None if no element + mass number is found.

    LLM usage: Level 4 (Substantial).
    """
    stem = Path(path).stem
    massFirst = re.match(r"(\d+)([A-Za-z]{1,2})(?![a-z])", stem)
    elementFirst = re.match(r"([A-Za-z]{1,2})(\d+)", stem)
    if massFirst:
        massNumber, element = massFirst.group(1), massFirst.group(2)
    elif elementFirst:
        element, massNumber = elementFirst.group(1), elementFirst.group(2)
    else:
        return None
    return element.capitalize() + massNumber


def targetLabel(targetKey):
    """LaTeX-style label like $^{27}$Al for a target key like "Al27".

    LLM usage: Level 4 (Substantial).
    """
    if targetKey in TARGETS:
        return TARGETS[targetKey]["label"]
    match = re.match(r"([A-Za-z]+)(\d+)", targetKey or "")
    return rf"$^{{{match.group(2)}}}${match.group(1)}" if match else str(targetKey)


def readMeasuredSpectra(path, histogramName=None):
    """Read the per-detector LaBr3 energy spectra (TH2: energy x detector) with uproot.

    returns: dict with binEdges (bins + 1,), binCentres (bins,), counts (detectors, bins),
        histogramName

    LLM usage: Level 4 (Substantial).
    """
    with uproot.open(path) as rootFile:
        available = {key.split(";")[0] for key in rootFile.keys()}
        candidates = [histogramName] if histogramName else HISTOGRAM_NAMES
        found = [name for name in candidates if name in available]
        if not found:
            raise KeyError(f"None of {candidates} in {path}")
        histogram = rootFile[found[0]]
        return {
            "binEdges": np.asarray(histogram.axis(0).edges(), dtype=float),
            "binCentres": np.asarray(histogram.axis(0).centers(), dtype=float),
            "counts": np.asarray(histogram.values(), dtype=float).T,
            "histogramName": found[0],
        }


def estimateThreshold(keVBinCentres, measuredCounts, keVSearchMax=300.0):
    """Energy where the smoothed measured counts first reach half their maximum below keVSearchMax.

    keVSearchMax stays below the 511 keV peak, which would otherwise set the maximum (106Pd).

    LLM usage: Level 4 (Substantial).
    """
    region = keVBinCentres < keVSearchMax
    smoothed = gaussian_filter1d(measuredCounts[region].astype(float), 3)
    return float(keVBinCentres[region][np.argmax(smoothed >= 0.5 * smoothed.max())])


def fitPeaksAndBackground(keVBinCentres, measuredCounts, keVPeakEnergies, peakTypes, inputs,
                          backgroundCoefficients, lowEnergyRatio, keVThreshold, keVUpperCutoff, keVFitMin=120.0):
    """Peak areas and background level from one weighted non-negative least-squares fit.

    Templates (on the measured binning, counts per bin): for each peak its full response
    (full-energy peak + Compton continuum + escape peaks) for area 1, and the 28Si background
    shape (sums to 1); all multiplied by the threshold and upper cut-off. Rows are weighted by
    1 / sqrt(counts) (Poisson), so the low-energy bins with many counts do not dominate.
    Overlapping peaks (e.g. 28Si 6879 and 16O 6917 keV) share the counts, but how they are
    split is poorly determined.

    returns:
        peakAreas: (peaks,) full-energy peak areas (counts before threshold/cut-off)
        backgroundAmplitude: number to multiply the background shape by
        response: (bins,) threshold x cut-off on the measured binning

    LLM usage: Level 4 (Substantial).
    """
    response = (sim.lowEnergyThreshold(keVBinCentres, keVThreshold, 8.0)
                * (1 - sim.lowEnergyThreshold(keVBinCentres, keVUpperCutoff, 100.0)))
    peakTemplates = [
        sim.peakResponse(keVBinCentres, keVEnergy, 1.0, inputs["detector"], inputs["escapeRatios"],
                         peakType=peakType)[0] * response
        for keVEnergy, peakType in zip(keVPeakEnergies, peakTypes)
    ]
    backgroundTemplate = sim.backgroundShape(
        keVBinCentres, backgroundCoefficients, inputs["keVFitMin"], inputs["keVFitMax"],
        lowEnergyRatio=lowEnergyRatio, keVLowGrid=inputs["keVLowGrid"], humpScale=1.0) * response
    templates = np.column_stack(peakTemplates + [backgroundTemplate])

    inFitRange = (keVBinCentres >= keVFitMin) & (keVBinCentres <= keVUpperCutoff)
    weights = 1 / np.sqrt(np.maximum(measuredCounts[inFitRange], 1.0))
    amplitudes, _ = nnls(templates[inFitRange] * weights[:, None], measuredCounts[inFitRange] * weights)
    return amplitudes[:-1], amplitudes[-1], response


def compareDataset(datasetPath, detector, args):
    """Fit, simulate and plot one measured dataset. Returns the path of the saved figure.

    LLM usage: Level 4 (Substantial).
    """
    targetKey = args.target or targetFromFilename(datasetPath)
    target = TARGETS.get(targetKey, {})
    keVUpperCutoff = args.upper_cutoff or target.get("keVUpperCutoff", 10000.0)
    keVPeakEnergies = list(args.peaks) if args.peaks else list(target.get("peaks", []))
    if not keVPeakEnergies:
        raise ValueError(f"No peak list for target {targetKey!r}; pass --peaks or add it to TARGETS")

    measured = readMeasuredSpectra(datasetPath, args.histogram)
    if not 0 <= detector < measured["counts"].shape[0]:
        raise ValueError(f"Detector {detector} out of range (file has {measured['counts'].shape[0]})")
    if measured["counts"][detector].sum() == 0:
        raise ValueError(f"Detector {detector} is empty in {datasetPath.name}")

    # Keep only the useful energy range, so the simulation grid stays small
    numberOfBins = int(np.searchsorted(measured["binEdges"], keVUpperCutoff + 1500.0))
    binEdges = measured["binEdges"][:numberOfBins + 1]
    binCentres = measured["binCentres"][:numberOfBins]
    measuredCounts = measured["counts"][detector][:numberOfBins]
    keVBinWidth = binEdges[1] - binEdges[0]

    settings = dict(sim.DEFAULT_SETTINGS)
    settings["numberOfChannels"] = numberOfBins
    settings["keVGridMax"] = float(np.ceil(binEdges[-1]))
    settings["minResponseFactorForLabel"] = 0.0
    inputs = sim.loadSimulationInputs(settings)

    peakTypes = [sim.PEAK_TYPES["fullEnergy"]] * len(keVPeakEnergies)
    if not args.no_fixed_peaks:
        keVPeakEnergies += [sim.KEV_ANNIHILATION_PEAK, sim.KEV_LA138_PEAK]
        peakTypes += [sim.PEAK_TYPES["annihilation511"], sim.PEAK_TYPES["La138"]]

    backgroundCoefficients = inputs["backgroundCoefficients"][args.background_detector].copy()
    lowEnergyRatio = inputs["lowEnergyRatios"][args.background_detector]  # measured hump shape, humpScale 1
    keVThreshold = estimateThreshold(binCentres, measuredCounts)
    peakAreas, backgroundAmplitude, response = fitPeaksAndBackground(
        binCentres, measuredCounts, keVPeakEnergies, peakTypes, inputs, backgroundCoefficients, lowEnergyRatio,
        keVThreshold, keVUpperCutoff)

    # Totals of the fitted model after threshold and cut-off, to set totalCounts and backgroundFraction
    peakTotal = sum(
        (sim.peakResponse(binCentres, keVEnergy, area, inputs["detector"], inputs["escapeRatios"],
                          peakType=peakType)[0] * response).sum()
        for keVEnergy, area, peakType in zip(keVPeakEnergies, peakAreas, peakTypes)
    )
    backgroundTotal = backgroundAmplitude * (
        sim.backgroundShape(binCentres, backgroundCoefficients, inputs["keVFitMin"], inputs["keVFitMax"],
                            lowEnergyRatio=lowEnergyRatio, keVLowGrid=inputs["keVLowGrid"], humpScale=1.0)
        * response).sum()
    numberOfPeaks = len(keVPeakEnergies)
    parameters = {
        "numberOfRandomPeaks": 0,
        "keVPeakEnergies": np.asarray(keVPeakEnergies, dtype=float),
        "peakAreas": np.maximum(np.asarray(peakAreas, dtype=float), 1e-9),
        "peakTypes": np.asarray(peakTypes, dtype=int),
        "widthMultipliers": np.ones(numberOfPeaks),
        "comptonRatioFactors": np.ones(numberOfPeaks),
        "escapeRatioFactors": np.ones(numberOfPeaks),
        "keVOffset": float(binEdges[0]), "keVPerChannel": float(keVBinWidth), "keVPerChannel2": 0.0,
        "fwhmScale": 1.0,
        "backgroundDetectorIndex": args.background_detector,
        "backgroundCoefficients": backgroundCoefficients,
        "humpScale": 1.0,
        "keVThreshold": keVThreshold, "keVThresholdWidth": 8.0,
        "keVUpperCutoff": keVUpperCutoff, "keVUpperCutoffWidth": 100.0,
        "totalCounts": float(peakTotal + backgroundTotal),
        "backgroundFraction": float(backgroundTotal / max(peakTotal + backgroundTotal, 1e-300)),
    }
    expectedCounts, labels = sim.expectedSpectrum(parameters, inputs, settings)
    simulatedCounts = np.random.default_rng(args.seed).poisson(expectedCounts)

    datasetName = datasetPath.name
    print(f"\n{datasetName} ({targetKey}), detector {detector}, histogram {measured['histogramName']}, "
          f"{keVBinWidth:g} keV bins")
    print(f"  measured counts {measuredCounts.sum():.4g}, simulated {simulatedCounts.sum():.4g}, "
          f"threshold {keVThreshold:.0f} keV, background fraction {parameters['backgroundFraction']:.2f}")
    for keVEnergy, area in zip(keVPeakEnergies, peakAreas):
        print(f"  peak {keVEnergy:7.1f} keV: fitted area {area:9.0f}")

    def smooth(values):
        return gaussian_filter1d(values.astype(float), args.smooth) if args.smooth > 0 else values

    # Display histograms: sum groups of bins into wider bins (counts are summed, not averaged)
    binsPerDisplayBin = max(1, int(round(args.display_bin / keVBinWidth)))
    numberOfDisplayBins = binCentres.size // binsPerDisplayBin
    usedBins = numberOfDisplayBins * binsPerDisplayBin
    displayEdges = binEdges[:usedBins + 1:binsPerDisplayBin]
    displayCentres = 0.5 * (displayEdges[1:] + displayEdges[:-1])
    displayMeasured = smooth(measuredCounts)[:usedBins].reshape(numberOfDisplayBins, binsPerDisplayBin).sum(axis=1)
    displaySimulated = smooth(simulatedCounts)[:usedBins].reshape(numberOfDisplayBins, binsPerDisplayBin).sum(axis=1)
    keVDisplayBinWidth = binsPerDisplayBin * keVBinWidth

    figure, axes = plt.subplots(2, 1, figsize=(13, 8.5))
    regions = [(0.0, keVUpperCutoff + 500.0), target.get("keVZoom", (300.0, 3500.0))]
    typeNames = {code: name for name, code in sim.PEAK_TYPES.items()}
    yMax = 3 * displayMeasured.max()
    for axis, (keVLow, keVHigh) in zip(axes, regions):
        # Histograms: one step per display bin, drawn between the bin edges
        binIndices = np.flatnonzero((displayCentres >= keVLow) & (displayCentres <= keVHigh))
        edgesInRange = displayEdges[binIndices[0]:binIndices[-1] + 2]
        axis.stairs(displayMeasured[binIndices], edgesInRange, lw=0.8,
                    label=f"measured: {datasetName}, detector {detector}")
        axis.stairs(displaySimulated[binIndices], edgesInRange, lw=0.8, alpha=0.85, label="simulated")
        for code in np.unique(labels["peakType"]):
            selected = ((labels["peakType"] == code) & (labels["keVEnergy"] >= keVLow)
                        & (labels["keVEnergy"] <= keVHigh))
            axis.vlines(labels["keVEnergy"][selected], 0.5, yMax, colors=f"C{code + 2}",
                        lw=0.6, alpha=0.6, linestyles="dashed", label=f"label: {typeNames[code]}")
        axis.set_yscale("log")
        axis.set_xlim(keVLow, keVHigh)
        axis.set_ylim(0.5, yMax)
        axis.set_xlabel("Energy [keV]")
        axis.set_ylabel(f"counts per {keVDisplayBinWidth:g} keV")
    axes[0].set_title(f"{targetLabel(targetKey)}: measured ({datasetName}, detector {detector}) vs simulated\n"
                      f"(known peaks, fitted areas, background shape from 28Si detector "
                      f"{args.background_detector}, no random factors)")
    axes[0].legend(fontsize=7, ncol=2)
    figure.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outputPath = FIG_DIR / f"compare_simulation_{targetKey or datasetPath.stem}_detector{detector}.png"
    figure.savefig(outputPath, dpi=150)
    plt.close(figure)
    print(f"  saved {outputPath}")
    return outputPath


def resolveDatasets(dataset):
    """Paths for the dataset argument: a path, a file name in data/measured, or "all".

    LLM usage: Level 4 (Substantial).
    """
    if dataset == "all":
        return sorted(MEASURED_DIR.glob("*.root"))
    path = Path(dataset)
    if not path.exists():
        path = MEASURED_DIR / dataset
    if not path.exists():
        raise FileNotFoundError(f"{dataset} not found (also looked in {MEASURED_DIR})")
    return [path]


def main() -> None:
    """Compare one or all measured datasets with the simulator.

    LLM usage: Level 4 (Substantial).
    """
    argumentParser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    argumentParser.add_argument("dataset", nargs="?", default=DEFAULT_DATASET,
                                help="file in data/measured, a path, or 'all' (default: the 28Si run)")
    argumentParser.add_argument("--detector", type=int, default=0, help="measured detector index")
    argumentParser.add_argument("--target", default=None,
                                help=f"target key {list(TARGETS)} (default: read from the file name)")
    argumentParser.add_argument("--peaks", type=float, nargs="+", default=None,
                                help="full-energy peak energies in keV (default: the target's list)")
    argumentParser.add_argument("--no-fixed-peaks", action="store_true",
                                help="leave out the 511 keV and La-138 (1470 keV) peaks")
    argumentParser.add_argument("--upper-cutoff", type=float, default=None,
                                help="keV where the measured spectrum ends (default: per target)")
    argumentParser.add_argument("--background-detector", type=int, default=0,
                                help="28Si detector whose fitted background shape is used")
    argumentParser.add_argument("--histogram", default=None,
                                help=f"histogram name (default: first of {HISTOGRAM_NAMES} in the file)")
    argumentParser.add_argument("--seed", type=int, default=2026)
    argumentParser.add_argument("--display-bin", type=float, default=8.0,
                                help="width (keV) of the histogram bins in the plot; measured bins are summed "
                                     "into these (plot only, the fit uses the original bins)")
    argumentParser.add_argument("--smooth", type=float, default=0.0,
                                help="optional Gaussian smoothing (original bins) before display binning; 0 = none")
    args = argumentParser.parse_args()

    for datasetPath in resolveDatasets(args.dataset):
        compareDataset(datasetPath, args.detector, args)


if __name__ == "__main__":
    main()
