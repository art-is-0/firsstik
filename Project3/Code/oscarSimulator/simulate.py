"""Simulate labelled LaBr3 gamma spectra (in channels) for training the peak finder.

Pipeline for one spectrum (all randomness is drawn first, the rest is deterministic):
    1. drawSpectrumParameters: peak energies and areas, calibration, widths, background,
       threshold, total counts ... (uses the random generator; nothing else does)
    2. peakResponse (per full-energy peak): Gaussian full-energy peak + Compton continuum +
       single/double escape peaks, as expected counts per keV-grid bin
    3. expectedSpectrum: sum of all peak responses + background, times the low-energy
       threshold and upper cut-off, rebinned to channels; labels = peak positions in channels
    4. simulateSpectrum: one Poisson draw on the expected counts per channel
    5. makeDataset / saveDataset: many spectra with padded label arrays, saved to data/datasets/

Labels are stored as peak positions (fractional channels) plus metadata, not as per-channel
target arrays, so the label design (centre channel or window) can be chosen later.

Use one generator per data split, spawned once:
    trainingSeed, validationSeed, testSeed = np.random.SeedSequence(2026).spawn(3)

LLM usage: Level 4 (Substantial); an LLM wrote the structure, signatures and docstrings of this
file and most function bodies. The author wrote parts of loadSimulationInputs and
drawSpectrumParameters. Each function's docstring states its level.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy.special import erf

from data_reader import DATASETS_DIR, read_escape_ratios
from detector import LaBr3Detector
from spectrum_parts import (
    backgroundShape,
    channelEdgesInKeV,
    comptonContinuumShape,
    comptonEdgeEnergy, 
    escapePeakEnergies,
    fitBackgroundCoefficients,
    gaussianPeak,
    lowEnergyBackgroundRatios,
    keVToChannel,
    lowEnergyThreshold,
    rebinToChannels,
)

from pathlib import Path

from data_reader import (PROJECT_DIR,
                         FIG_DIR,
                         DATA_DIR,
                         REFERENCE_DIR,
                         MEASURED_DIR,
                         BACKGROUNDS_DIR,
                         DATASETS_DIR,
                         DEFAULT_BACKGROUND_PATH,
                         DEFAULT_ESCAPE_RATIOS_PATH,
                         DEFAULT_HTML_PATH)



# Peak types stored in the labels (integers, so they fit in a NumPy array)
PEAK_TYPES = {"fullEnergy": 0, "singleEscape": 1, "doubleEscape": 2, "La138": 3, "annihilation511": 4}

KEV_ANNIHILATION_PEAK = 511.0
KEV_LA138_PEAK = 1470.0  # 1436 keV gamma summed with the Ba K X-ray (about 32-38 keV)

# Fraction of a Gaussian's area within +-1 FWHM of its centre (+-2.355 sigma), for the significance
FRACTION_WITHIN_ONE_FWHM = erf(2 * np.sqrt(2 * np.log(2)) / np.sqrt(2))

# Ranges for the random parameters. Placeholder values: set them from the real data
# (gain range per detector, counts per spectrum, ...) and widen them (randomise rather than imitate).
DEFAULT_SETTINGS = {
    "numberOfChannels": 8192,              # must match the network input made from real spectra
    "keVGridStep": 1.0,                    # fine simulation grid, finer than one channel
    "keVGridMax": 20000.0,
    "numberOfPeaksRange": (1, 40),         # random full-energy peaks per spectrum
    "keVLowPeakEnergyRange": (100.0, 6000.0),     # most peaks: discrete low-lying transitions, log-uniform
    "keVHighPeakEnergyRange": (6000.0, 12000.0),  # rare high-energy peaks (primaries), uniform
    "fractionHighEnergyPeaks": 0.15,              # so escape peaks and high-energy peaks are seen
    "peakAreaRange": (1e2, 1e5),           # counts in a full-energy peak (draw log-uniformly)
    "totalCountsRange": (1e5, 1e7),
    "backgroundFractionRange": (0.3, 0.95),
    "keVOffsetRange": (-50.0, 50.0),
    "keVPerChannelRange": (1.0, 2.5),      # gain
    "keVPerChannel2Range": (-2e-6, 2e-6),  # small quadratic term
    "fwhmScaleRange": (0.85, 1.2),         # whole FWHM curve scaled per spectrum
    "widthMultiplierRange": (1.0, 2.0),    # extra broadening for some peaks (Doppler)
    "fractionBroadenedPeaks": 0.2,
    "ratioRandomFactor": 2.0,              # Compton/escape ratios multiplied by up to x/÷ this
    "humpScaleRange": (0.5, 2.0),          # x the measured low-energy hump (below 500 keV)
    "keVThresholdRange": (40.0, 120.0),
    "keVThresholdWidthRange": (3.0, 15.0),
    "keVUpperCutoffRange": (8000.0, 16000.0),
    "keVUpperCutoffWidthRange": (50.0, 300.0),
    "backgroundSlopeNoisePerMeV": 0.05,         # noise added to the fitted ln B coefficients
    "backgroundCurvatureNoisePerMeV2": 0.003,
    "includeLa138": True,                  # intrinsic peak at about 1470 keV (1436 + Ba K X-ray)
    "includeAnnihilation511": True,
    "minResponseFactorForLabel": 0.5,      # drop labels where threshold x cut-off is below this
    "maxPeaksPerSpectrum": 200,            # padding length for the label arrays
}


def logUniform(randomGenerator, low, high, size=None):
    """Numbers drawn uniformly in log space between low and high.

    LLM usage: Level 2 (Snippet).
    """
    return np.exp(randomGenerator.uniform(np.log(low), np.log(high), size=size))


def loadSimulationInputs(settings=DEFAULT_SETTINGS):
    """Load everything that is the same for all spectra, once.

    returns: dict with
        detector: LaBr3Detector (FWHM curve, efficiency)
        escapeRatios: dict from read_escape_ratios (SE/FE, DE/FE, FE/interacting vs energy)
        backgroundCoefficients, keVFitMin, keVFitMax: from fitBackgroundCoefficients
        lowEnergyRatios, keVLowGrid: measured background shape below keVFitMin, per detector
            (from lowEnergyBackgroundRatios)
        keVGrid, keVGridEdges: centres and edges of the fine keV simulation grid

    LLM usage: Level 3 (Skeleton).
    """
    detector = LaBr3Detector()
    # Read the escape-peak table written by photon_transport_mc.py
    if not DEFAULT_ESCAPE_RATIOS_PATH.exists():
        raise FileNotFoundError(
            f"{DEFAULT_ESCAPE_RATIOS_PATH} not found. Create it with python photon_transport_mc.py"
        )
    else:
        escapeRatios = read_escape_ratios(DEFAULT_ESCAPE_RATIOS_PATH)

    backgroundCoefficients, keVFitMin, keVFitMax = fitBackgroundCoefficients(DEFAULT_BACKGROUND_PATH)
    keVLowGrid, lowEnergyRatios = lowEnergyBackgroundRatios(DEFAULT_BACKGROUND_PATH, keVJoin=keVFitMin)

    keVGridEdges = np.arange(0, settings["keVGridMax"] + settings["keVGridStep"], settings["keVGridStep"])
    keVGrid = 0.5 * (keVGridEdges[1:] + keVGridEdges[:-1])
    
    inputs = {'detector':detector,
             'escapeRatios':escapeRatios,
             'backgroundCoefficients':backgroundCoefficients,
             'keVFitMin':keVFitMin,
             'keVFitMax':keVFitMax,
             'lowEnergyRatios':lowEnergyRatios,
             'keVLowGrid':keVLowGrid,
             'keVGrid':keVGrid,
             'keVGridEdges':keVGridEdges}
    
    return inputs



def drawSpectrumParameters(randomGenerator, inputs, settings=DEFAULT_SETTINGS):
    """Draw all random parameters of one spectrum. The only place that uses randomness.

    Draws (suggested; per spectrum unless noted):
        - number of full-energy peaks; their energies; their areas (log-uniform)
        - per peak: width multiplier (1 for most peaks), Compton and escape ratio factors
        - fixed peaks if switched on (La138 at ~1470 keV, external 511 keV) with random areas
        - energy-to-channel response: keVOffset, keVPerChannel, keVPerChannel2
        - FWHM scale for the whole spectrum
        - background: index of a detector's coefficients (+ small noise), strength of the
          low-energy hump (humpScale), background fraction of the total counts
        - threshold position and width, upper cut-off energy
        - total counts (difficulty knob)

    returns: dict of plain numbers and arrays, enough to rebuild the spectrum exactly

    Peak areas are relative: expectedSpectrum rescales them so that the spectrum has
    totalCounts counts, of which the fraction backgroundFraction is background.

    LLM usage: Level 4 (Substantial); the author wrote the peak energy/area draws, an LLM the rest.
    """
    numberOfPeaks = randomGenerator.integers(*settings["numberOfPeaksRange"], endpoint=True)
    # Peak energies: mostly log-uniform below 6 MeV (where resolved peaks usually are), plus a
    # small fraction uniform at high energy so the network also sees escape peaks and primaries.
    keVLowMin, keVLowMax = settings["keVLowPeakEnergyRange"]
    keVHighMin, keVHighMax = settings["keVHighPeakEnergyRange"]
    isHighEnergy = randomGenerator.random(numberOfPeaks) < settings["fractionHighEnergyPeaks"]
    keVPeakEnergies = np.where(
        isHighEnergy,
        randomGenerator.uniform(keVHighMin, keVHighMax, size=numberOfPeaks),
        np.exp(randomGenerator.uniform(np.log(keVLowMin), np.log(keVLowMax), size=numberOfPeaks)),
    )
    peakAreas = np.exp(randomGenerator.uniform(np.log(settings['peakAreaRange'][0]), np.log(settings['peakAreaRange'][1]), size=numberOfPeaks))
    peakTypes = np.full(numberOfPeaks, PEAK_TYPES["fullEnergy"])

    # Fixed peaks present in (almost) every OSCAR spectrum: external 511 keV and intrinsic La-138
    fixedPeakEnergies, fixedPeakTypes = [], []
    if settings["includeAnnihilation511"]:
        fixedPeakEnergies.append(KEV_ANNIHILATION_PEAK)
        fixedPeakTypes.append(PEAK_TYPES["annihilation511"])
    if settings["includeLa138"]:
        fixedPeakEnergies.append(KEV_LA138_PEAK)
        fixedPeakTypes.append(PEAK_TYPES["La138"])
    fixedPeakAreas = logUniform(randomGenerator, *settings["peakAreaRange"], size=len(fixedPeakEnergies))

    keVPeakEnergies = np.concatenate((keVPeakEnergies, fixedPeakEnergies))
    peakAreas = np.concatenate((peakAreas, fixedPeakAreas))
    peakTypes = np.concatenate((peakTypes, fixedPeakTypes)).astype(int)
    totalNumberOfPeaks = keVPeakEnergies.size

    # Per peak: extra broadening for some peaks (Doppler, unresolved doublets), and random factors
    # on the Compton and escape ratios from the photon-transport table
    isBroadened = randomGenerator.random(totalNumberOfPeaks) < settings["fractionBroadenedPeaks"]
    widthMultipliers = np.where(
        isBroadened, randomGenerator.uniform(*settings["widthMultiplierRange"], size=totalNumberOfPeaks), 1.0
    )
    ratioFactor = settings["ratioRandomFactor"]
    comptonRatioFactors = logUniform(randomGenerator, 1 / ratioFactor, ratioFactor, size=totalNumberOfPeaks)
    escapeRatioFactors = logUniform(randomGenerator, 1 / ratioFactor, ratioFactor, size=totalNumberOfPeaks)

    # Energy-to-channel response of the simulated detector (unknown gain, offset, non-linearity)
    keVOffset = randomGenerator.uniform(*settings['keVOffsetRange'])
    keVPerChannel = randomGenerator.uniform(*settings['keVPerChannelRange'])
    keVPerChannel2 = randomGenerator.uniform(*settings['keVPerChannel2Range'])

    fwhmScale = randomGenerator.uniform(*settings['fwhmScaleRange'])

    # Background: one measured detector's fitted shape, with small random changes in slope and
    # curvature (coefficients are in np.polyval order, so [-2] is the linear and [-3] the quadratic term)
    backgroundDetectorIndex = randomGenerator.integers(inputs["backgroundCoefficients"].shape[0])
    backgroundCoefficients = inputs["backgroundCoefficients"][backgroundDetectorIndex].copy()
    backgroundCoefficients[-2] += randomGenerator.normal(0, settings["backgroundSlopeNoisePerMeV"])
    backgroundCoefficients[-3] += randomGenerator.normal(0, settings["backgroundCurvatureNoisePerMeV2"])
    humpScale = randomGenerator.uniform(*settings["humpScaleRange"])  # strength of the low-energy hump
    backgroundFraction = randomGenerator.uniform(*settings["backgroundFractionRange"])

    keVThreshold = randomGenerator.uniform(*settings["keVThresholdRange"])
    keVThresholdWidth = randomGenerator.uniform(*settings["keVThresholdWidthRange"])
    keVUpperCutoff = randomGenerator.uniform(*settings["keVUpperCutoffRange"])
    keVUpperCutoffWidth = randomGenerator.uniform(*settings["keVUpperCutoffWidthRange"])

    totalCounts = logUniform(randomGenerator, *settings["totalCountsRange"])

    return {
        "numberOfRandomPeaks": int(numberOfPeaks),
        "keVPeakEnergies": keVPeakEnergies,
        "peakAreas": peakAreas,
        "peakTypes": peakTypes,
        "widthMultipliers": widthMultipliers,
        "comptonRatioFactors": comptonRatioFactors,
        "escapeRatioFactors": escapeRatioFactors,
        "keVOffset": keVOffset,
        "keVPerChannel": keVPerChannel,
        "keVPerChannel2": keVPerChannel2,
        "fwhmScale": fwhmScale,
        "backgroundDetectorIndex": int(backgroundDetectorIndex),
        "backgroundCoefficients": backgroundCoefficients,
        "humpScale": humpScale,
        "backgroundFraction": backgroundFraction,
        "keVThreshold": keVThreshold,
        "keVThresholdWidth": keVThresholdWidth,
        "keVUpperCutoff": keVUpperCutoff,
        "keVUpperCutoffWidth": keVUpperCutoffWidth,
        "totalCounts": totalCounts,
    }


def peakResponse(keVGrid, keVPeakEnergy, peakArea, detector, escapeRatios,
                 fwhmScale=1.0, widthMultiplier=1.0, comptonRatioFactor=1.0, escapeRatioFactor=1.0,
                 peakType=PEAK_TYPES["fullEnergy"]):
    """Expected counts per keV-grid bin from one full-energy peak, plus the peaks to label.

    Contents:
        - full-energy peak: gaussianPeak(keVGrid, E, FWHM(E), peakArea)
        - Compton continuum: comptonContinuumShape(keVGrid, E, FWHM(E_C)) * peakArea * comptonToPeak,
          with comptonToPeak ≈ 1 / fe_over_interacting - 1 - se_over_fe - de_over_fe (interpolated
          at E from escapeRatios), times comptonRatioFactor
        - escape peaks (only where escapePeakEnergies is finite): gaussianPeak at E - 511 and
          E - 1022 keV with FWHM at those energies, areas peakArea * se_over_fe (de_over_fe) *
          escapeRatioFactor
    FWHM(E) = detector.fwhm(E) * fwhmScale * widthMultiplier.

    peakType is the label type of the full-energy peak itself (fullEnergy, La138 or
    annihilation511); its escape peaks are labelled singleEscape and doubleEscape.

    returns:
        expectedCounts: (len(keVGrid),) expected counts per grid bin
        labelledPeaks: list of (keVEnergy, peakType, area, keVFWHM) for FE, SE, DE
            (not the Compton edge: it is not a peak)

    LLM usage: Level 4 (Substantial).
    """
    def keVFwhmAt(keVEnergy):
        return detector.fwhm(keVEnergy) * fwhmScale * widthMultiplier

    # Ratios from the photon-transport table, interpolated at this energy
    tableEnergies = escapeRatios["energy_keV"]
    seOverFe = np.interp(keVPeakEnergy, tableEnergies, escapeRatios["se_over_fe"])
    deOverFe = np.interp(keVPeakEnergy, tableEnergies, escapeRatios["de_over_fe"])
    feOverInteracting = np.interp(keVPeakEnergy, tableEnergies, escapeRatios["fe_over_interacting"])
    # Everything that interacts but is not in FE, SE or DE ends up in the continuum
    comptonToPeak = max(1 / feOverInteracting - 1 - seOverFe - deOverFe, 0.0) * comptonRatioFactor

    keVFwhm = keVFwhmAt(keVPeakEnergy)
    expectedCounts = gaussianPeak(keVGrid, keVPeakEnergy, keVFwhm, peakArea)
    keVComptonFwhm = keVFwhmAt(comptonEdgeEnergy(keVPeakEnergy))
    expectedCounts = expectedCounts + (
        comptonContinuumShape(keVGrid, keVPeakEnergy, keVComptonFwhm) * peakArea * comptonToPeak
    )
    labelledPeaks = [(keVPeakEnergy, peakType, peakArea, keVFwhm)]

    keVSingleEscape, keVDoubleEscape = escapePeakEnergies(keVPeakEnergy)
    if np.isfinite(keVSingleEscape):
        for keVEscape, escapeOverFe, escapeType in (
            (float(keVSingleEscape), seOverFe, PEAK_TYPES["singleEscape"]),
            (float(keVDoubleEscape), deOverFe, PEAK_TYPES["doubleEscape"]),
        ):
            escapeArea = peakArea * escapeOverFe * escapeRatioFactor
            if escapeArea <= 0:
                continue
            keVEscapeFwhm = keVFwhmAt(keVEscape)
            expectedCounts = expectedCounts + gaussianPeak(keVGrid, keVEscape, keVEscapeFwhm, escapeArea)
            labelledPeaks.append((keVEscape, escapeType, escapeArea, keVEscapeFwhm))

    return expectedCounts, labelledPeaks


def expectedSpectrum(parameters, inputs, settings=DEFAULT_SETTINGS):
    """Deterministic expected spectrum in channels and its labels, from drawn parameters.

    Steps:
        1. sum peakResponse over all peaks (random and fixed ones)
        2. add backgroundShape(...) * backgroundCounts
        3. multiply everything by lowEnergyThreshold and by the upper cut-off
           (1 - lowEnergyThreshold(keVGrid, keVUpperCutoff, width))
        4. channelEdgesInKeV with the drawn calibration; rebinToChannels
        5. labels: keVToChannel for all labelled peaks; drop peaks outside the channel range
           or where the threshold factor is small (e.g. < 0.5)

    Peak areas in the parameters are relative: the peak part is scaled to
    (1 - backgroundFraction) * totalCounts and the background to backgroundFraction * totalCounts
    (counts after threshold and cut-off, over the whole keV grid). Label areas are the scaled ones.

    returns:
        expectedCountsPerChannel: (numberOfChannels,)
        labels: dict of arrays, one entry per kept peak:
            channel: fractional channel coordinate (channel i covers [i, i + 1), centre i + 0.5)
            peakType, keVEnergy, area (expected counts), fwhmChannels
            significance: 0.98 * area / sqrt(all expected counts within +-1 FWHM); filter on
                this later to ignore peaks that are invisible under the background

    LLM usage: Level 4 (Substantial).
    """
    keVGrid, keVGridEdges = inputs["keVGrid"], inputs["keVGridEdges"]
    detector, escapeRatios = inputs["detector"], inputs["escapeRatios"]

    # 1. all peaks with their Compton continua and escape peaks (relative areas)
    peaksPerGridBin = np.zeros_like(keVGrid)
    labelledPeaks = []
    for i in range(parameters["keVPeakEnergies"].size):
        peakCounts, peaks = peakResponse(
            keVGrid, parameters["keVPeakEnergies"][i], parameters["peakAreas"][i], detector, escapeRatios,
            fwhmScale=parameters["fwhmScale"], widthMultiplier=parameters["widthMultipliers"][i],
            comptonRatioFactor=parameters["comptonRatioFactors"][i],
            escapeRatioFactor=parameters["escapeRatioFactors"][i], peakType=parameters["peakTypes"][i],
        )
        peaksPerGridBin += peakCounts
        labelledPeaks.extend(peaks)

    # 2. background shape (sums to 1)
    backgroundPerGridBin = backgroundShape(
        keVGrid, parameters["backgroundCoefficients"], inputs["keVFitMin"], inputs["keVFitMax"],
        lowEnergyRatio=inputs["lowEnergyRatios"][parameters["backgroundDetectorIndex"]],
        keVLowGrid=inputs["keVLowGrid"], humpScale=parameters["humpScale"],
    )

    # 3. detector threshold and upper cut-off act on everything
    def responseFactor(keVEnergy):
        return (lowEnergyThreshold(keVEnergy, parameters["keVThreshold"], parameters["keVThresholdWidth"])
                * (1 - lowEnergyThreshold(keVEnergy, parameters["keVUpperCutoff"], parameters["keVUpperCutoffWidth"])))

    gridResponse = responseFactor(keVGrid)
    peaksPerGridBin *= gridResponse
    backgroundPerGridBin *= gridResponse

    totalCounts, backgroundFraction = parameters["totalCounts"], parameters["backgroundFraction"]
    peakScale = (1 - backgroundFraction) * totalCounts / max(peaksPerGridBin.sum(), 1e-300)
    backgroundScale = backgroundFraction * totalCounts / max(backgroundPerGridBin.sum(), 1e-300)
    expectedPerGridBin = peakScale * peaksPerGridBin + backgroundScale * backgroundPerGridBin

    # 4. onto channels
    numberOfChannels = settings["numberOfChannels"]
    keVChannelEdges = channelEdgesInKeV(
        numberOfChannels, parameters["keVOffset"], parameters["keVPerChannel"], parameters["keVPerChannel2"]
    )
    expectedCountsPerChannel = rebinToChannels(keVGridEdges, expectedPerGridBin, keVChannelEdges)

    # 5. labels in channels
    keVLabelEnergies = np.array([peak[0] for peak in labelledPeaks], dtype=float)
    labelTypes = np.array([peak[1] for peak in labelledPeaks], dtype=int)
    labelResponse = responseFactor(keVLabelEnergies)
    labelAreas = np.array([peak[2] for peak in labelledPeaks], dtype=float) * peakScale * labelResponse
    keVLabelFwhm = np.array([peak[3] for peak in labelledPeaks], dtype=float)

    labelChannels = keVToChannel(keVLabelEnergies, keVChannelEdges)
    keep = ((keVLabelEnergies > keVChannelEdges[0]) & (keVLabelEnergies < keVChannelEdges[-1])
            & (labelResponse >= settings["minResponseFactorForLabel"]))

    # FWHM in channels with the local gain dE/dch at the peak
    localKeVPerChannel = parameters["keVPerChannel"] + 2 * parameters["keVPerChannel2"] * labelChannels
    fwhmChannels = keVLabelFwhm / localKeVPerChannel

    # Significance: peak counts within +-1 FWHM over the square root of all counts there
    cumulativeCounts = np.concatenate(([0.0], np.cumsum(expectedCountsPerChannel)))
    channelEdgeCoordinates = np.arange(numberOfChannels + 1)
    countsInWindow = (np.interp(labelChannels + fwhmChannels, channelEdgeCoordinates, cumulativeCounts)
                      - np.interp(labelChannels - fwhmChannels, channelEdgeCoordinates, cumulativeCounts))
    significance = FRACTION_WITHIN_ONE_FWHM * labelAreas / np.sqrt(np.maximum(countsInWindow, 1e-12))

    labels = {
        "channel": labelChannels[keep],
        "peakType": labelTypes[keep],
        "keVEnergy": keVLabelEnergies[keep],
        "area": labelAreas[keep],
        "fwhmChannels": fwhmChannels[keep],
        "significance": significance[keep],
    }
    return expectedCountsPerChannel, labels


def simulateSpectrum(randomGenerator, inputs, settings=DEFAULT_SETTINGS):
    """Draw parameters, build the expected spectrum, and do one Poisson draw.

    Poisson sampling happens once, on the total expected counts per channel; never add
    Gaussian noise afterwards.

    returns:
        counts: (numberOfChannels,) integer counts
        labels: as in expectedSpectrum
        parameters: as in drawSpectrumParameters (kept for analysis, e.g. recall vs counts)

    LLM usage: Level 4 (Substantial).
    """
    parameters = drawSpectrumParameters(randomGenerator, inputs, settings)
    expectedCountsPerChannel, labels = expectedSpectrum(parameters, inputs, settings)
    counts = randomGenerator.poisson(expectedCountsPerChannel)
    return counts, labels, parameters


def makeDataset(numberOfSpectra, randomGenerator, inputs, settings=DEFAULT_SETTINGS):
    """Simulate many spectra and pack them into fixed-size arrays.

    returns: dict with
        counts: (numberOfSpectra, numberOfChannels), e.g. uint32
        labelChannels, labelPeakTypes, labelAreas, labelFwhmChannels:
            (numberOfSpectra, maxPeaksPerSpectrum), padded with NaN (-1 for peakTypes)
        numberOfLabels: (numberOfSpectra,)
        selected parameters per spectrum (total counts, gain, ...) for later analysis

    Label arrays are named "label" + the key in expectedSpectrum's labels, e.g. labelChannel,
    labelPeakType, labelSignificance.

    LLM usage: Level 4 (Substantial).
    """
    numberOfChannels = settings["numberOfChannels"]
    maxPeaks = settings["maxPeaksPerSpectrum"]
    labelKeys = ["channel", "peakType", "keVEnergy", "area", "fwhmChannels", "significance"]
    parameterKeys = ["totalCounts", "backgroundFraction", "keVOffset", "keVPerChannel", "keVPerChannel2",
                     "fwhmScale", "numberOfRandomPeaks", "backgroundDetectorIndex", "keVThreshold",
                     "keVUpperCutoff", "humpScale"]

    dataset = {"counts": np.zeros((numberOfSpectra, numberOfChannels), dtype=np.uint32),
               "numberOfLabels": np.zeros(numberOfSpectra, dtype=int)}
    for key in labelKeys:
        labelName = "label" + key[0].upper() + key[1:]
        if key == "peakType":
            dataset[labelName] = np.full((numberOfSpectra, maxPeaks), -1, dtype=int)
        else:
            dataset[labelName] = np.full((numberOfSpectra, maxPeaks), np.nan)
    for key in parameterKeys:
        dataset[key] = np.zeros(numberOfSpectra)

    for spectrumIndex in range(numberOfSpectra):
        counts, labels, parameters = simulateSpectrum(randomGenerator, inputs, settings)
        numberOfLabels = labels["channel"].size
        if numberOfLabels > maxPeaks:
            raise ValueError(f"{numberOfLabels} labels > maxPeaksPerSpectrum = {maxPeaks}; increase it")

        dataset["counts"][spectrumIndex] = counts
        dataset["numberOfLabels"][spectrumIndex] = numberOfLabels
        for key in labelKeys:
            dataset["label" + key[0].upper() + key[1:]][spectrumIndex, :numberOfLabels] = labels[key]
        for key in parameterKeys:
            dataset[key][spectrumIndex] = parameters[key]
    return dataset


def saveDataset(dataset, name, settings=DEFAULT_SETTINGS, seedInfo=""):
    """Save a dataset to DATASETS_DIR / f"{name}.npz" together with the settings and seed,
    so that it can be regenerated exactly.

    The settings and PEAK_TYPES are stored as JSON strings ("settingsJson", "peakTypesJson");
    read them back with json.loads(str(data["settingsJson"])).

    returns: path of the written file

    LLM usage: Level 4 (Substantial).
    """
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    outputPath = DATASETS_DIR / f"{name}.npz"
    np.savez_compressed(
        outputPath, **dataset,
        settingsJson=np.array(json.dumps(settings)),
        peakTypesJson=np.array(json.dumps(PEAK_TYPES)),
        seedInfo=np.array(seedInfo),
    )
    return outputPath


def main() -> None:
    """Generate and save the training, validation and test sets.

    Spawn the three generators once from np.random.SeedSequence(2026) and give each split
    its own. Validation and test sets are saved; the training set may instead be generated
    on the fly during training (use --training 0 to skip it).

    These are synthetic validation/test sets for development. The final evaluation uses the
    hand-labelled real spectra.

    LLM usage: Level 4 (Substantial).
    """
    argumentParser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    argumentParser.add_argument("--training", type=int, default=2000, help="number of training spectra")
    argumentParser.add_argument("--validation", type=int, default=500, help="number of validation spectra")
    argumentParser.add_argument("--test", type=int, default=500, help="number of synthetic test spectra")
    argumentParser.add_argument("--seed", type=int, default=2026)
    args = argumentParser.parse_args()

    inputs = loadSimulationInputs(DEFAULT_SETTINGS)
    splitSeeds = np.random.SeedSequence(args.seed).spawn(3)  # order: training, validation, test
    splits = [("training", args.training), ("validation", args.validation), ("test", args.test)]

    for splitIndex, (splitName, numberOfSpectra) in enumerate(splits):
        if numberOfSpectra <= 0:
            continue
        randomGenerator = np.random.default_rng(splitSeeds[splitIndex])
        startTime = time.perf_counter()
        dataset = makeDataset(numberOfSpectra, randomGenerator, inputs, DEFAULT_SETTINGS)
        outputPath = saveDataset(dataset, splitName, DEFAULT_SETTINGS,
                                 seedInfo=f"np.random.SeedSequence({args.seed}).spawn(3)[{splitIndex}]")
        print(f"{splitName}: {numberOfSpectra} spectra in {time.perf_counter() - startTime:.1f} s "
              f"-> {outputPath}")


if __name__ == "__main__":
    main()
