# oscarSimulator

A simulator of labelled gamma-ray spectra for the LaBr₃(Ce) detectors of the OSCAR array at the
Oslo Cyclotron Laboratory (30 detectors, 3.5" × 8" crystals). It produces synthetic spectra in
**uncalibrated channel space** together with the exact positions of all peaks, to be used as
training data for machine-learning peak finding. The peak-finding models themselves are not part
of this folder yet.

Each simulated spectrum contains:
- a random number of full-energy peaks at random energies, plus the 511 keV annihilation peak and
  the intrinsic La-138 peak (~1470 keV)
- for every peak, its Compton continuum (Klein–Nishina) and its single- and double-escape peaks
- a smooth background whose shape comes from measured OSCAR spectra, including the low-energy hump
- a low-energy threshold, an upper cut-off, a random energy-to-channel response (gain, offset,
  non-linearity) and Poisson noise

Everything that varies between real spectra is randomised ("randomise rather than imitate"), so
that a model trained on the simulated spectra does not depend on one detector or one calibration.

## Installation

Python 3.12 was used. Install the dependencies (ROOT is **not** needed; `.root` files are read with
uproot):

```bash
pip install -r requirements.txt
```

Run all commands from inside this folder.

## Quick start: generate a dataset

The repository already contains the two precomputed inputs the simulator needs
(`data/reference/escape_ratios_LaBr3.csv` and `data/backgrounds/backgrounds_si28-…npz`), so you can
generate spectra straight away:

```bash
python simulate.py                       # 2000 training, 500 validation, 500 test spectra
python simulate.py --training 0          # only validation and test
python simulate.py --training 10000 --validation 1000 --test 1000 --seed 2026
```

This writes `data/datasets/training.npz`, `validation.npz` and `test.npz` (about 0.04 s per
spectrum). The three splits use independent random streams from `np.random.SeedSequence(seed).spawn(3)`,
so the same seed always gives the same data.

**Before generating data for real use**, check the ranges in `DEFAULT_SETTINGS` at the top of
`simulate.py`. In particular `numberOfChannels` and `keVPerChannelRange` must match how the real
spectra are binned when they are given to the model.

To load a dataset:

```python
import json, numpy as np
data = np.load("data/datasets/validation.npz")
counts = data["counts"]                     # (N spectra, channels): the model input
n = data["numberOfLabels"][0]               # number of peaks in spectrum 0
peakChannels = data["labelChannel"][0, :n]  # their positions (fractional channels)
peakTypes = json.loads(str(data["peakTypesJson"]))
```

The labels (`labelChannel`, `labelPeakType`, `labelKeVEnergy`, `labelArea`, `labelFwhmChannels`,
`labelSignificance`) and the stored per-spectrum parameters are explained in
`spectrum_generator.ipynb`, section "The labels in a simulated dataset". In short:
`labelSignificance` = 0.98 · area / √(all counts within ±1 FWHM), so you can keep only peaks that
are visible above the noise, e.g. `labelSignificance > 3`.

## Full pipeline (regenerating the inputs)

Only needed if you change the detector geometry or want backgrounds from other measurements.

```
NIST XCOM table ──► photon_transport_mc.py ──► data/reference/escape_ratios_LaBr3.csv ─┐
                                                                                        ├─► simulate.py ──► data/datasets/*.npz
measured .root  ──► extract_backgrounds.py ──► data/backgrounds/backgrounds_<run>.npz ─┘
```

1. **Escape-peak table** (a few minutes):
   ```bash
   python photon_transport_mc.py            # 1 M photons per energy, 100 keV – 20 MeV
   python photon_transport_mc.py --check    # verify the Klein–Nishina sampler only
   ```
2. **Background shapes** from a measured run (needs the `.root` file in `data/measured/`):
   ```bash
   python extract_backgrounds.py data/measured/si28-sirius-20190313-142346.root --plot-detector 0
   ```
3. **Simulate** as in the quick start.
4. **Check the simulator against measured spectra** (optional):
   ```bash
   python compare_simulation.py all                         # every file in data/measured
   python compare_simulation.py 27Al_cal2.root --detector 3
   ```
   Figures go to `figures/compare_simulation_<target>_detector<N>.png`.

The measured `.root` files are not in the repository (they are large and git-ignored). Put them
in `data/measured/`. `extract_backgrounds.py` expects the calibrated histogram `energy_cal_labr`
(energy × detector); `compare_simulation.py` also reads `h_LaBr3_energy`.

## What each file does

| File | Purpose |
|---|---|
| `simulate.py` | The simulator. Draws all random parameters of a spectrum, builds the expected spectrum, rebins it to channels, does one Poisson draw, and writes datasets with labels. Run it as a script (see above) or import its functions. |
| `spectrum_parts.py` | Deterministic building blocks: Gaussian peaks, Compton edge and continuum, escape-peak energies, the background model (fit + measured low-energy shape), the threshold, and the mapping from keV to channels (`channelEdgesInKeV`, `keVToChannel`, `rebinToChannels`). |
| `detector.py` | `LaBr3Detector`: crystal geometry and density, linear attenuation coefficients from XCOM (log-log interpolation), the FWHM curve, an efficiency curve, and a sampler of first-interaction points. |
| `data_reader.py` | All file paths (`DATA_DIR`, `REFERENCE_DIR`, …) and readers for the XCOM table (`read_xcom_table`) and the escape-ratio table (`read_escape_ratios`). |
| `photon_transport_mc.py` | Toy photon-transport Monte Carlo for one crystal (photoelectric, Compton, pair production, both 511 keV photons tracked). Writes the single/double-escape-to-full-energy ratios and peak-to-total versus energy. |
| `background_estimation.py` | `snipBackground`: the SNIP background-estimation algorithm (identical to ROOT's `TSpectrum::Background`, plus an energy-dependent window). |
| `extract_backgrounds.py` | Reads measured per-detector spectra from a `.root` file and saves their SNIP backgrounds, which the background model is fitted to. |
| `compare_simulation.py` | Simulates a spectrum with known peak energies for a measured target (28Si, 27Al, 106Pd), with areas and background level fitted to the measurement, and plots both for a visual check. |
| `spectrum_generator.ipynb` | Development notebook: physics checks, how to make and read datasets, explanation of the labels, comparison figures. |
| `requirements.txt` | Python dependencies. |
| `UsesOfGenerativeAI.md` | Declaration of generative-AI use per file (course requirement). |
| `CLAUDE.md`, `Handoff …md` | Design notes and background for the project. |

### Data and output folders

| Folder | Contents |
|---|---|
| `data/reference/` | Physics inputs: NIST XCOM attenuation table for LaBr₃, escape-ratio table |
| `data/backgrounds/` | SNIP backgrounds extracted from measured runs (`.npz`) |
| `data/measured/` | Measured OSCAR `.root` files (not in git) |
| `data/datasets/` | Simulated datasets (not in git; regenerate with `simulate.py`) |
| `figures/` | Plots made by the scripts |

## Main assumptions and limitations

- Peak shapes are Gaussian with the measured OSCAR FWHM curve (randomly scaled and broadened).
- Compton continua use single-scatter Klein–Nishina; escape ratios come from a toy Monte Carlo of
  a bare crystal (no housing or neighbours, no bremsstrahlung escape).
- The background shape comes from one measured run (28Si), with random variations. Other reactions
  have different backgrounds, mostly at high energy; this is covered only by the randomisation.
- The La-138 beta shelf (789–1044 keV) is not included.

## Reproducibility

All randomness goes through one `numpy.random.Generator` per data split, created once from a seed.
The settings, peak-type codes and seed are stored inside every dataset file.
