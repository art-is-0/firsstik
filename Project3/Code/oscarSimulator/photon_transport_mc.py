"""Toy photon-transport Monte Carlo for one LaBr3 crystal.

Purpose: estimate the single-escape (SE) and double-escape (DE) peak areas
relative to the full-energy (FE) peak, plus the peak-to-total ratio, instead of
guessing them. Each photon is followed through the crystal until it is absorbed
or leaves, including Compton-scattered photons and both 511 keV annihilation
photons. The peak areas are then counted from the total deposited energy per
emitted gamma.

Geometry: bare cylinder (radius R, length L) with its axis along z, front face
at z = 0. Point source on the axis at z = -D, emitting isotropically into the
cone that hits the front face.

Physics per interaction (Knoll ch. 2 and 10; Leo ch. 2):
    - Free path from total attenuation *without* coherent scattering.
    - Process chosen from mu_photo, mu_Compton (XCOM "incoherent") and mu_pair.
    - Photoelectric: all energy deposited locally.
    - Compton: angle sampled from Klein-Nishina; electron energy deposited locally.
    - Pair: E - 1022 keV deposited locally; two back-to-back 511 keV photons
      start at the same point in an isotropic direction.

Assumptions / limitations (state these in the report):
    - Bare crystal: no housing, reflector, shielding or neighbouring detectors.
    - Electrons and positrons deposit all their kinetic energy where they are
      created; bremsstrahlung and positron annihilation in flight are ignored.
    - Coherent (Rayleigh) scattering is ignored; it deposits no energy and only
      slightly changes direction.
    - La/Br K X-ray escape after photoabsorption is ignored (a small
      ~30-40 keV effect).
    - Ce (~5%) is treated as La; XCOM coefficients are for pure LaBr3.
    - No energy resolution: peak areas are counted from exact deposits.

Run:
    ~/machinelearning/bin/python photon_transport_mc.py      # full grid -> data/reference/escape_ratios_LaBr3.csv
    ~/machinelearning/bin/python photon_transport_mc.py --photons 200000 --energies 2000 4000 --output ''
    ~/machinelearning/bin/python photon_transport_mc.py --check   # verify the Compton sampler

LLM usage: Level 4 (Substantial); an LLM wrote this script.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from data_reader import DEFAULT_ESCAPE_RATIOS_PATH
from detector import LaBr3Detector

ELECTRON_REST_ENERGY_KEV = 511.0
CUTOFF_ENERGY_KEV = 1.0  # photons below this deposit their energy and stop
PEAK_TOLERANCE_KEV = 0.5  # deposits within this of E, E-511, E-1022 count as peak


def distanceToCylinderSurface(cmPositions, directions, cmRadius, cmLength):
    """Distance (cm) from points inside the cylinder to its surface along each direction.

    params:
        cmPositions: (n, 3) array of points inside the crystal, cm
        directions: (n, 3) array of unit vectors
        cmRadius, cmLength: crystal dimensions, cm

    returns:
        (n,) array of distances to the side wall or the end faces, whichever is closer

    LLM usage: Level 4 (Substantial).
    """
    x, y, z = cmPositions.T
    ux, uy, uz = directions.T

    # Side wall: solve |(x, y) + t (ux, uy)| = R for the positive root
    a = ux**2 + uy**2
    b = x * ux + y * uy
    c = x**2 + y**2 - cmRadius**2  # <= 0 inside the crystal
    with np.errstate(divide="ignore", invalid="ignore"):
        cmToSideWall = np.where(
            a > 1e-12, (-b + np.sqrt(np.maximum(b**2 - a * c, 0.0))) / a, np.inf
        )
        cmToEndFace = np.where(
            uz > 1e-12, (cmLength - z) / uz, np.where(uz < -1e-12, -z / uz, np.inf)
        )
    return np.minimum(cmToSideWall, cmToEndFace)


def isotropicDirections(count, randomGenerator):
    """Unit vectors uniformly distributed over the sphere.

    LLM usage: Level 4 (Substantial).
    """
    cosTheta = 2 * randomGenerator.random(count) - 1
    radiansPhi = 2 * np.pi * randomGenerator.random(count)
    sinTheta = np.sqrt(1 - cosTheta**2)
    return np.column_stack(
        (sinTheta * np.cos(radiansPhi), sinTheta * np.sin(radiansPhi), cosTheta)
    )


def rotateDirections(directions, cosTheta, radiansPhi):
    """Turn each unit vector by polar angle theta and azimuth phi around itself.

    Standard formula used in photon-transport codes (e.g. EGS, MCNP).

    LLM usage: Level 4 (Substantial).
    """
    ux, uy, uz = directions.T
    sinTheta = np.sqrt(np.maximum(1 - cosTheta**2, 0.0))
    cosPhi, sinPhi = np.cos(radiansPhi), np.sin(radiansPhi)
    sinPolar = np.sqrt(np.maximum(1 - uz**2, 0.0))

    nearlyAlongZ = sinPolar < 1e-8
    safeSinPolar = np.where(nearlyAlongZ, 1.0, sinPolar)
    newX = sinTheta * (ux * uz * cosPhi - uy * sinPhi) / safeSinPolar + ux * cosTheta
    newY = sinTheta * (uy * uz * cosPhi + ux * sinPhi) / safeSinPolar + uy * cosTheta
    newZ = -sinTheta * cosPhi * sinPolar + uz * cosTheta

    # For directions along +-z the formula is singular; build the vector directly.
    newX = np.where(nearlyAlongZ, sinTheta * cosPhi, newX)
    newY = np.where(nearlyAlongZ, sinTheta * sinPhi, newY)
    newZ = np.where(nearlyAlongZ, np.sign(uz) * cosTheta, newZ)
    newDirections = np.column_stack((newX, newY, newZ))
    return newDirections / np.linalg.norm(newDirections, axis=1, keepdims=True)


def sampleKleinNishinaCosTheta(keVEnergies, randomGenerator):
    """Sample Compton scattering angles from the Klein-Nishina cross section.

    dsigma/dOmega is proportional to f = P^2 (P + 1/P - sin^2 theta) / 2 with
    P = E'/E = 1 / (1 + alpha (1 - cos theta)). f is at most 1 (at theta = 0),
    so cos theta is drawn uniformly on [-1, 1] and accepted with probability f.

    LLM usage: Level 4 (Substantial).
    """
    alpha = keVEnergies / ELECTRON_REST_ENERGY_KEV
    cosTheta = np.empty_like(keVEnergies)
    stillToSample = np.ones(keVEnergies.size, dtype=bool)
    while stillToSample.any():
        indices = np.flatnonzero(stillToSample)
        candidate = 2 * randomGenerator.random(indices.size) - 1
        ratio = 1 / (1 + alpha[indices] * (1 - candidate))  # P = E'/E
        acceptance = 0.5 * ratio**2 * (ratio + 1 / ratio - (1 - candidate**2))
        accepted = randomGenerator.random(indices.size) < acceptance
        cosTheta[indices[accepted]] = candidate[accepted]
        stillToSample[indices[accepted]] = False
    return cosTheta


def simulateDepositedEnergies(keVPeakEnergy, numberOfPhotons, detector, randomGenerator):
    """Total energy deposited in the crystal by each emitted gamma that hits the front face.

    params:
        keVPeakEnergy: energy of the emitted gamma (the full-energy peak), keV
        numberOfPhotons: number of gammas aimed at the front face
        detector: LaBr3Detector (geometry, density, XCOM interpolation)
        randomGenerator: np.random.Generator, created once by the caller

    returns:
        keVDeposited: (numberOfPhotons,) total deposited energy per gamma (0 if it passed through)
        solidAngleFraction: fraction of 4 pi covered by the front face, so that
            absolute efficiency = (counts in peak / numberOfPhotons) * solidAngleFraction

    LLM usage: Level 4 (Substantial).
    """
    cmRadius, cmLength = detector.cmRadius, detector.cmLength
    cmDistance = detector.cmDistanceToSource
    attenuation = detector.interpolatedLinearAttinuation

    # Point source on the axis: directions uniform in solid angle inside the cone that hits the face
    cosThetaMax = cmDistance / np.hypot(cmDistance, cmRadius)
    solidAngleFraction = (1 - cosThetaMax) / 2
    cosTheta = 1 - (1 - cosThetaMax) * randomGenerator.random(numberOfPhotons)
    sinTheta = np.sqrt(1 - cosTheta**2)
    radiansPhi = 2 * np.pi * randomGenerator.random(numberOfPhotons)
    directions = np.column_stack(
        (sinTheta * np.cos(radiansPhi), sinTheta * np.sin(radiansPhi), cosTheta)
    )
    cmRadiusOnFace = cmDistance * sinTheta / cosTheta
    cmPositions = np.column_stack(
        (cmRadiusOnFace * np.cos(radiansPhi), cmRadiusOnFace * np.sin(radiansPhi),
         np.zeros(numberOfPhotons))
    )

    keVEnergies = np.full(numberOfPhotons, float(keVPeakEnergy))
    eventIndex = np.arange(numberOfPhotons)  # which emitted gamma each tracked photon belongs to
    keVDeposited = np.zeros(numberOfPhotons)

    while keVEnergies.size > 0:
        # Free path from total attenuation without coherent scattering
        muPhoto = attenuation(keVEnergies, "photoelectric_cm2_g")
        muCompton = attenuation(keVEnergies, "incoherent_scattering_cm2_g")
        muPair = (attenuation(keVEnergies, "pair_nuclear_cm2_g")
                  + attenuation(keVEnergies, "pair_electron_cm2_g"))
        muTotal = muPhoto + muCompton + muPair

        cmFreePath = -np.log(1 - randomGenerator.random(keVEnergies.size)) / muTotal
        cmToSurface = distanceToCylinderSurface(cmPositions, directions, cmRadius, cmLength)
        interacts = cmFreePath < cmToSurface

        # Photons that leave are dropped; the rest move to their interaction point
        keVEnergies, eventIndex = keVEnergies[interacts], eventIndex[interacts]
        cmPositions = cmPositions[interacts] + cmFreePath[interacts, None] * directions[interacts]
        directions = directions[interacts]
        muPhoto, muCompton, muTotal = muPhoto[interacts], muCompton[interacts], muTotal[interacts]

        # Choose the process
        uniformProcess = randomGenerator.random(keVEnergies.size) * muTotal
        isPhoto = uniformProcess < muPhoto
        isCompton = ~isPhoto & (uniformProcess < muPhoto + muCompton)
        isPair = ~isPhoto & ~isCompton

        # Photoelectric: deposit everything
        np.add.at(keVDeposited, eventIndex[isPhoto], keVEnergies[isPhoto])

        # Compton: deposit the electron energy, continue with the scattered photon
        keVBefore = keVEnergies[isCompton]
        cosScatter = sampleKleinNishinaCosTheta(keVBefore, randomGenerator)
        keVAfter = keVBefore / (1 + keVBefore / ELECTRON_REST_ENERGY_KEV * (1 - cosScatter))
        np.add.at(keVDeposited, eventIndex[isCompton], keVBefore - keVAfter)
        comptonDirections = rotateDirections(
            directions[isCompton], cosScatter,
            2 * np.pi * randomGenerator.random(keVBefore.size),
        )

        # Pair: deposit E - 1022 keV, create two back-to-back 511 keV photons
        np.add.at(keVDeposited, eventIndex[isPair],
                  keVEnergies[isPair] - 2 * ELECTRON_REST_ENERGY_KEV)
        numberOfPairs = int(isPair.sum())
        annihilationDirections = isotropicDirections(numberOfPairs, randomGenerator)

        # Photons still alive: scattered Compton photons + both annihilation photons
        keVEnergies = np.concatenate(
            (keVAfter, np.full(2 * numberOfPairs, ELECTRON_REST_ENERGY_KEV))
        )
        eventIndex = np.concatenate(
            (eventIndex[isCompton], eventIndex[isPair], eventIndex[isPair])
        )
        cmPositions = np.concatenate(
            (cmPositions[isCompton], cmPositions[isPair], cmPositions[isPair])
        )
        directions = np.concatenate(
            (comptonDirections, annihilationDirections, -annihilationDirections)
        )

        # Very low-energy photons are absorbed on the spot
        belowCutoff = keVEnergies < CUTOFF_ENERGY_KEV
        np.add.at(keVDeposited, eventIndex[belowCutoff], keVEnergies[belowCutoff])
        keep = ~belowCutoff
        keVEnergies, eventIndex = keVEnergies[keep], eventIndex[keep]
        cmPositions, directions = cmPositions[keep], directions[keep]

    return keVDeposited, solidAngleFraction


def peakAreas(keVDeposited, keVPeakEnergy):
    """Count events in the full-energy, single-escape and double-escape peaks.

    Without resolution smearing every peak event deposits exactly E, E - 511 or
    E - 1022 keV, while continuum events are spread out, so a narrow tolerance
    picks the peaks cleanly.

    returns:
        dict with counts in "FE", "SE", "DE" and "interacting" (deposit > 0)

    LLM usage: Level 4 (Substantial).
    """
    def countNear(keVTarget):
        return int(np.sum(np.abs(keVDeposited - keVTarget) < PEAK_TOLERANCE_KEV))

    hasEscapePeaks = keVPeakEnergy > 2 * ELECTRON_REST_ENERGY_KEV
    return {
        "FE": countNear(keVPeakEnergy),
        "SE": countNear(keVPeakEnergy - ELECTRON_REST_ENERGY_KEV) if hasEscapePeaks else 0,
        "DE": countNear(keVPeakEnergy - 2 * ELECTRON_REST_ENERGY_KEV) if hasEscapePeaks else 0,
        "interacting": int(np.sum(keVDeposited > 0)),
    }


def ratioWithError(numerator, denominator):
    """Ratio of two Poisson counts with its 1-sigma error (independent counts approximation).

    LLM usage: Level 4 (Substantial).
    """
    if numerator == 0 or denominator == 0:
        return 0.0, 0.0
    ratio = numerator / denominator
    return ratio, ratio * np.sqrt(1 / numerator + 1 / denominator)


def checkKleinNishinaSampler(randomGenerator, keVEnergy=662.0, numberOfSamples=1_000_000):
    """Verification: compare sampled Compton electron energies with the analytic dsigma/dT.

    The analytic shape is the same Klein-Nishina expression as comptonContinuumShape
    (Knoll ch. 10). Prints the largest relative deviation over the histogram bins.

    LLM usage: Level 4 (Substantial).
    """
    alpha = keVEnergy / ELECTRON_REST_ENERGY_KEV
    cosTheta = sampleKleinNishinaCosTheta(np.full(numberOfSamples, keVEnergy), randomGenerator)
    keVElectron = keVEnergy - keVEnergy / (1 + alpha * (1 - cosTheta))
    keVEdge = keVEnergy * 2 * alpha / (1 + 2 * alpha)

    binEdges = np.linspace(0, keVEdge, 41)
    counts, _ = np.histogram(keVElectron, bins=binEdges)
    binCentres = 0.5 * (binEdges[1:] + binEdges[:-1])
    s = binCentres / keVEnergy
    analytic = 2 + s**2 / (alpha**2 * (1 - s)**2) + s / (1 - s) * (s - 2 / alpha)
    expected = analytic / analytic.sum() * counts.sum()

    relativeDeviation = np.abs(counts - expected) / expected
    statisticalError = 1 / np.sqrt(expected)
    print(f"Klein-Nishina check at {keVEnergy:g} keV, {numberOfSamples} samples:")
    print(f"  largest relative deviation {relativeDeviation.max():.4f}, "
          f"typical statistical error {statisticalError.mean():.4f}")
    print(f"  largest deviation in units of sigma: {(relativeDeviation / statisticalError).max():.2f}")


DEFAULT_OUTPUT_PATH = DEFAULT_ESCAPE_RATIOS_PATH

CSV_COLUMNS = [
    "energy_keV",
    "se_over_fe", "se_over_fe_error",
    "de_over_fe", "de_over_fe_error",
    "fe_over_interacting",
    "fe_absolute_efficiency",
    "photons_simulated", "counts_interacting", "counts_fe", "counts_se", "counts_de",
]


def defaultEnergyGrid():
    """Full-energy peak energies (keV) for the stored table: sparse below 1022 keV, dense above.

    Escape peaks only exist above 1022 keV and change quickly just above it,
    so that region gets 100 keV steps; 200 keV steps above 3 MeV.

    LLM usage: Level 4 (Substantial).
    """
    belowPairThreshold = np.array([100, 200, 300, 400, 500, 662, 800, 1000])
    nearThreshold = np.arange(1100, 3000, 100)
    highEnergy = np.arange(3000, 20001, 200)
    return np.concatenate((belowPairThreshold, nearThreshold, highEnergy)).astype(float)


def computeEscapeTableRow(keVPeakEnergy, numberOfPhotons, detector, randomGenerator):
    """Simulate one full-energy peak energy and return its row of CSV_COLUMNS values.

    LLM usage: Level 4 (Substantial).
    """
    keVDeposited, solidAngleFraction = simulateDepositedEnergies(
        keVPeakEnergy, numberOfPhotons, detector, randomGenerator
    )
    areas = peakAreas(keVDeposited, keVPeakEnergy)
    singleEscapeRatio, singleEscapeError = ratioWithError(areas["SE"], areas["FE"])
    doubleEscapeRatio, doubleEscapeError = ratioWithError(areas["DE"], areas["FE"])
    return [
        keVPeakEnergy,
        singleEscapeRatio, singleEscapeError,
        doubleEscapeRatio, doubleEscapeError,
        areas["FE"] / areas["interacting"],
        areas["FE"] / numberOfPhotons * solidAngleFraction,
        numberOfPhotons, areas["interacting"], areas["FE"], areas["SE"], areas["DE"],
    ]


def writeEscapeTable(outputPath, rows, detector, numberOfPhotons, seed):
    """Write the table as CSV: '#' metadata lines, one header row, then one row per energy.

    LLM usage: Level 4 (Substantial).
    """
    metadataLines = [
        "Escape-peak ratios for one bare LaBr3 crystal, from photon_transport_mc.py",
        f"crystal radius {detector.cmRadius:.4f} cm, length {detector.cmLength:.2f} cm, "
        f"density {detector.g_cm3Density} g/cm3, on-axis point source "
        f"{detector.cmDistanceToSource:g} cm in front of the face",
        f"{numberOfPhotons} photons aimed at the front face per energy, seed {seed}",
        "se_over_fe, de_over_fe: single/double-escape peak area divided by full-energy peak area "
        "(errors are 1 sigma, Poisson)",
        "fe_over_interacting: full-energy peak counts / gammas that deposited any energy "
        "(peak-to-total)",
        "fe_absolute_efficiency: full-energy peak counts per gamma emitted into 4 pi",
        "Limitations: bare crystal, local electron deposit (no bremsstrahlung escape), "
        "no X-ray escape, no coherent scattering",
    ]
    with open(outputPath, "w", newline="") as outputFile:
        for line in metadataLines:
            outputFile.write(f"# {line}\n")
        writer = csv.writer(outputFile)
        writer.writerow(CSV_COLUMNS)
        for row in rows:
            writer.writerow([f"{value:.6g}" if isinstance(value, float) else value
                             for value in row])


def main() -> None:
    """Simulate a grid of full-energy peak energies, print the ratios and write them to a CSV file.

    LLM usage: Level 4 (Substantial).
    """
    argumentParser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    argumentParser.add_argument("--photons", type=int, default=1_000_000,
                                help="gammas aimed at the front face per energy")
    argumentParser.add_argument("--energies", type=float, nargs="+", default=None,
                                help="full-energy peak energies in keV (default: 100 keV to 20 MeV grid)")
    argumentParser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH),
                                help="CSV file to write (use --output '' to skip writing)")
    argumentParser.add_argument("--seed", type=int, default=2026)
    argumentParser.add_argument("--check", action="store_true",
                                help="only run the Klein-Nishina sampler check")
    args = argumentParser.parse_args()

    randomGenerator = np.random.default_rng(args.seed)
    detector = LaBr3Detector(randomGenerator)

    if args.check:
        checkKleinNishinaSampler(randomGenerator)
        return

    keVPeakEnergies = defaultEnergyGrid() if args.energies is None else args.energies
    print(f"Crystal R = {detector.cmRadius:.3f} cm, L = {detector.cmLength:.2f} cm, "
          f"source {detector.cmDistanceToSource:g} cm in front, {args.photons} photons per energy")
    print(f"{'E (keV)':>8} {'SE/FE':>14} {'DE/FE':>14} {'FE/interacting':>15} {'FE abs. eff.':>13}")
    rows = []
    for keVPeakEnergy in keVPeakEnergies:
        row = computeEscapeTableRow(float(keVPeakEnergy), args.photons, detector, randomGenerator)
        rows.append(row)
        print(f"{row[0]:8.0f} {row[1]:7.3f}±{row[2]:5.3f} {row[3]:7.3f}±{row[4]:5.3f} "
              f"{row[5]:15.3f} {row[6]:13.2e}", flush=True)

    if args.output:
        writeEscapeTable(args.output, rows, detector, args.photons, args.seed)
        print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
