"""
Building blocks of the simulated spectra: peak shapes, Compton continua, escape-peak
energies, the smooth background, the low-energy threshold and the mapping from keV to
channels. Randomness is not added here; the simulation step draws the random parameters
and passes them in.

Convention: every function that returns part of a spectrum returns expected counts per
bin of the (evenly spaced) keV grid, or a shape that sums to 1 over that grid, so that the
parts can be added and then passed to rebinToChannels.
"""

from data_reader import DEFAULT_BACKGROUND_PATH
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.special import erf

ELECTRON_REST_ENERGY_KEV = 511.0


def gaussianPeak(keVGrid, keVPeakEnergy, keVFWHM, counts):
    """
    Expected counts per grid bin of a Gaussian peak at keVPeakEnergy with the given FWHM
    and total counts (area).

    keVGrid holds the centres of an evenly spaced grid. Each bin gets the integral of the
    Gaussian over the bin (difference of the Gaussian CDF at the bin edges), so the result
    is correct for any bin width, also when the bins are not much narrower than sigma.
    """
    keVGrid = np.asarray(keVGrid, dtype=float)
    sigma = keVFWHM / (2 * np.sqrt(2 * np.log(2)))  # LEO eq 4.19, p.87
    keVHalfBin = 0.5 * (keVGrid[1] - keVGrid[0])
    # Integral of the normalised Gaussian (LEO eq 4.20, p.87) from the lower to the upper bin edge
    upper = erf((keVGrid + keVHalfBin - keVPeakEnergy) / (np.sqrt(2) * sigma))
    lower = erf((keVGrid - keVHalfBin - keVPeakEnergy) / (np.sqrt(2) * sigma))
    return counts * 0.5 * (upper - lower)


def comptonEdgeEnergy(keVPeakEnergy):
    """Compton edge: the largest energy a single Compton scatter can give the electron
    (photon backscattered, theta = 180 degrees). Knoll ch. 10.
    """
    alpha = keVPeakEnergy / ELECTRON_REST_ENERGY_KEV  # photon energy in units of m_e c^2
    return keVPeakEnergy * 2 * alpha / (1 + 2 * alpha)


def comptonContinuumShape(keVGrid, keVPeakEnergy, keVFWHM):
    """Klein-Nishina electron-energy shape for one full-energy peak, smeared by resolution.
    keVGrid must be evenly spaced. Returns a shape that sums to 1 over the grid.

    To get expected counts, multiply by peak area (counts in gaussianPeak) x Compton-to-peak
    ratio (which we set ourselves).

    keVFWHM: the smearing width. The FWHM at the Compton edge, fwhm(comptonEdgeEnergy(E)),
    is the best single value, since the edge is the sharpest feature of the shape.

    LLM usage: Level 2 (Snippet); an LLM wrote the original version of this function.
    """
    keVGrid = np.asarray(keVGrid, dtype=float)
    alpha = keVPeakEnergy / ELECTRON_REST_ENERGY_KEV
    keVEdge = comptonEdgeEnergy(keVPeakEnergy)
    s = np.clip(keVGrid / keVPeakEnergy, 0, keVEdge / keVPeakEnergy)  # clipping keeps 1 - s > 0
    shape = 2 + s**2 / (alpha**2 * (1 - s)**2) + s / (1 - s) * (s - 2 / alpha)
    shape[keVGrid > keVEdge] = 0
    keVBinWidth = keVGrid[1] - keVGrid[0]
    shape = gaussian_filter1d(shape, (keVFWHM / 2.355) / keVBinWidth)
    total = shape.sum()
    return shape / total if total > 0 else shape  # all zero if the edge is below the grid


def escapePeakEnergies(keVPeakEnergy):
    """Single- and double-escape peak energies, E - 511 keV and E - 1022 keV.

    Escape peaks only exist when pair production is possible, so both are NaN for
    full-energy peaks at or below 1022 keV. Use np.isfinite to select the valid ones.
    """
    keVPeakEnergy = np.asarray(keVPeakEnergy, dtype=float)
    abovePairThreshold = keVPeakEnergy > 2 * ELECTRON_REST_ENERGY_KEV
    keVSingleEscapeEnergy = np.where(abovePairThreshold, keVPeakEnergy - ELECTRON_REST_ENERGY_KEV, np.nan)
    keVDoubleEscapeEnergy = np.where(abovePairThreshold, keVPeakEnergy - 2 * ELECTRON_REST_ENERGY_KEV, np.nan)
    return keVSingleEscapeEnergy, keVDoubleEscapeEnergy

def fitBackgroundCoefficients(filepath=DEFAULT_BACKGROUND_PATH, keVFitMin=500.0, keVFitMax=11000.0, degree=3):
    """Fit ln B(E) = polynomial in E (MeV) to every detector's SNIP background.

    params:
        filepath: .npz written by extract_backgrounds.py
        keVFitMin, keVFitMax: energy range of the fit, keV. Below about 500 keV the background
            rises steeply into the low-energy hump, which a low-order polynomial cannot follow
            (at 300 keV a cubic fitted from 300 keV was 65% too low); that region is described
            by lowEnergyBackgroundRatios instead. Above keVFitMax there are too few counts.
        degree: polynomial degree in ln B

    returns:
        coefficientsPerDetector: (detectors, degree + 1), highest power first (np.polyval order),
            for x = E / 1000 keV
        keVFitMin, keVFitMax: the fit range, needed to evaluate the shape safely
    """
    backgrounds = np.load(filepath)  # backgrounds of one dataset, one row per detector
    keVEnergy = backgrounds['keVBinCentres']
    fitRange = (keVEnergy > keVFitMin) & (keVEnergy < keVFitMax)

    coefficientsPerDetector = []
    for background in backgrounds['background']:
        usable = fitRange & (background > 0)  # log(0) is undefined
        coefficients = np.polyfit(keVEnergy[usable] / 1000, np.log(background[usable]), deg=degree)
        coefficientsPerDetector.append(coefficients)

    return np.array(coefficientsPerDetector), keVFitMin, keVFitMax


def lowEnergyBackgroundRatios(filepath=DEFAULT_BACKGROUND_PATH, keVJoin=500.0, keVLowestReliable=100.0):
    """Measured low-energy background shape of every detector, relative to its value at keVJoin.

    r_d(E) = SNIP_d(E) / SNIP_d(keVJoin) for E <= keVJoin, from the SNIP backgrounds of one run.
    It contains the low-energy hump at about 150-250 keV (photons backscattered in the material
    around the crystal, multiply scattered photons, the La-138 beta continuum). Below
    keVLowestReliable the measured spectrum is shaped by that run's own detector threshold, so r
    is held constant there; the simulator applies its own lowEnergyThreshold.

    returns:
        keVLowGrid: (n,) energies (the background file's bin centres up to keVJoin)
        ratios: (detectors, n), equal to 1 at keVJoin

    LLM usage: Level 2 (Snippet); an LLM wrote this function.
    """
    backgrounds = np.load(filepath)
    keVEnergy = backgrounds['keVBinCentres']
    lowRange = keVEnergy <= keVJoin
    keVLowGrid = keVEnergy[lowRange]

    ratios = []
    for background in backgrounds['background']:
        backgroundAtJoin = np.interp(keVJoin, keVEnergy, background)
        ratio = background[lowRange] / max(backgroundAtJoin, 1e-12)
        ratioAtLowestReliable = np.interp(keVLowestReliable, keVLowGrid, ratio)
        ratios.append(np.where(keVLowGrid < keVLowestReliable, ratioAtLowestReliable, ratio))
    return keVLowGrid, np.array(ratios)


def backgroundShape(keVGrid, coefficients, keVFitMin=500.0, keVFitMax=11000.0, slopeBelowFitPerMeV=0.0,
                    lowEnergyRatio=None, keVLowGrid=None, humpScale=1.0):
    """Smooth background on a keV grid from one set of fitted ln B coefficients. Not specific to any detector

    Three regions:
        - keVFitMin <= E <= keVFitMax: the fitted polynomial, B = exp(polyval(coefficients, E / 1000)).
        - E < keVFitMin, with a measured lowEnergyRatio r(E) (from lowEnergyBackgroundRatios):
          B(E) = B(keVFitMin) * (1 + humpScale * (r(E) - 1)). humpScale = 1 is the measured
          shape (including the hump), 0 is flat, larger values make the hump stronger.
          Without lowEnergyRatio: B(keVFitMin) * exp(slopeBelowFitPerMeV * (keVFitMin - E) / 1000).
        - E > keVFitMax: the fit continued as a straight line in ln B with its slope at
          keVFitMax, so the polynomial cannot bend upwards outside the fitted range.
    The pieces join continuously (r = 1 at keVFitMin). Apply lowEnergyThreshold to the full
    spectrum separately.

    returns:
        shape that sums to 1 over the grid (fraction of background counts per grid bin);
        multiply by the number of background counts

    Assumptions / limitations:
        - Below keVFitMin the shape is copied from one measured run (scaled by humpScale), not modelled.
        - Above keVFitMax the real spectrum ends where the reaction stops populating states;
          this depends on the reaction and is not modelled.

    LLM usage: Level 2 (Snippet); an LLM wrote this function.
    """
    keVGrid = np.asarray(keVGrid, dtype=float)
    logBackground = np.polyval(coefficients, keVGrid / 1000)

    logAtFitMin = np.polyval(coefficients, keVFitMin / 1000)
    below = keVGrid < keVFitMin
    if lowEnergyRatio is not None:
        ratio = np.interp(keVGrid[below], keVLowGrid, lowEnergyRatio)
        scaledRatio = np.maximum(1 + humpScale * (ratio - 1), 1e-12)
        logBackground[below] = logAtFitMin + np.log(scaledRatio)
    else:
        logBackground[below] = logAtFitMin + slopeBelowFitPerMeV * (keVFitMin - keVGrid[below]) / 1000

    logAtFitMax = np.polyval(coefficients, keVFitMax / 1000)
    slopeAtFitMaxPerMeV = np.polyval(np.polyder(coefficients), keVFitMax / 1000)
    above = keVGrid > keVFitMax
    logBackground[above] = logAtFitMax + slopeAtFitMaxPerMeV * (keVGrid[above] - keVFitMax) / 1000

    background = np.exp(logBackground)
    return background / background.sum()


def lowEnergyThreshold(keVGrid, keVThreshold, keVWidth):
    """Smooth detector threshold: about 0 well below keVThreshold, 1 well above it.

    A logistic step, 1 / (1 + exp(-(E - keVThreshold) / keVWidth)). Multiply the whole
    expected spectrum (peaks, Compton continua and background) by it, since the detector
    misses low-energy events of every kind. Measured spectra fall off below about
    50-120 keV, so draw keVThreshold around there and keVWidth from a few keV to ~15 keV.

    LLM usage: Level 2 (Snippet); an LLM wrote this function.
    """
    return 1 / (1 + np.exp(-(np.asarray(keVGrid, dtype=float) - keVThreshold) / keVWidth))


def channelEdgesInKeV(numberOfChannels, keVOffset, keVPerChannel, keVPerChannel2=0.0):
    """Energy (keV) at the edges of every channel for the calibration
    E(ch) = keVOffset + keVPerChannel * ch + keVPerChannel2 * ch**2.

    Channel i covers [edges[i], edges[i + 1]); its centre is at ch = i + 0.5.
    The calibration parameters are drawn at random in the simulation step, not here.

    params:
        numberOfChannels: length of the spectrum the network sees
        keVOffset: energy at the lower edge of channel 0, keV
        keVPerChannel: gain, keV per channel
        keVPerChannel2: small quadratic term, keV per channel^2

    returns:
        (numberOfChannels + 1,) array of increasing channel-edge energies, keV

    LLM usage: Level 2 (Snippet); an LLM wrote this function.
    """
    channelEdges = np.arange(numberOfChannels + 1)
    keVChannelEdges = keVOffset + keVPerChannel * channelEdges + keVPerChannel2 * channelEdges**2
    if np.any(np.diff(keVChannelEdges) <= 0):
        raise ValueError("Calibration is not increasing over the channel range; "
                         "make the quadratic term smaller")
    return keVChannelEdges


def keVToChannel(keVEnergy, keVChannelEdges):
    """Fractional channel number of each energy, for the peak labels.

    Interpolates in the channel-edge table, so a quadratic calibration needs no
    special treatment. Energies outside the channel range are clamped to its ends;
    drop those peaks from the labels.

    LLM usage: Level 2 (Snippet); an LLM wrote this function.
    """
    return np.interp(keVEnergy, keVChannelEdges, np.arange(keVChannelEdges.size))


def rebinToChannels(keVGridEdges, expectedCountsPerGridBin, keVChannelEdges):
    """Move expected counts from the fine keV grid onto the channels, conserving counts.

    Each channel gets the integral of the expected counts over its energy range:
    the cumulative counts are interpolated at the channel edges and differenced.
    Do this before the Poisson draw. The keV grid must be finer than the channels.
    Channels outside the keV grid get 0 counts.

    params:
        keVGridEdges: (n + 1,) edges of the fine simulation grid, keV
        expectedCountsPerGridBin: (n,) expected counts in each grid bin
        keVChannelEdges: (numberOfChannels + 1,) from channelEdgesInKeV

    returns:
        (numberOfChannels,) expected counts per channel

    LLM usage: Level 2 (Snippet); an LLM wrote this function.
    """
    cumulativeCounts = np.concatenate(([0.0], np.cumsum(expectedCountsPerGridBin)))
    return np.diff(np.interp(keVChannelEdges, keVGridEdges, cumulativeCounts))



