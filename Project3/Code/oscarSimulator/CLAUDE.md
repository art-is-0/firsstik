# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

FYS-STK4155 Project 3 (part of a course repo; sibling dirs `../week34`–`../week38` are weekly exercises). Deadline **11 December 2026**; report about 5000 words with 10–15 figures.

**Goal:** machine-learning peak finding in gamma spectra from the OSCAR array at OCL (30 LaBr₃(Ce) detectors, 3.5" × 8" crystals, density 5.08 g/cm³). Find *all* peak positions in channel space (about 8192 channels, to be confirmed). Calibration is done afterwards and is out of scope. The models are trained on **synthetic spectra** from the generator in this directory and tested on a small set of real, hand-labeled spectra.

The full design discussion is in `Handoff ML Peak Finding in LaBr₃ Gamma Spectra (OSCAR, OCL).md`. Read it before making design decisions. This is a custom project that the course staff still have to approve (fallback: Path B, Fashion-MNIST).

There is no build, lint, or test setup yet. Dependencies are in `requirements.txt` (numpy, scipy, matplotlib, uproot, ipykernel); ROOT is not required. The system `python3` lacks numpy, so use the venv at `~/machinelearning` (e.g. `~/machinelearning/bin/python`).

## How to work with the user

- The learning goal is understanding. **Explain the concepts and ask before writing large pieces of code.** Prefer small, testable functions over a finished pipeline. When reviewing, list bugs and explain the fixes rather than silently rewriting the user's code.
- Report text: only spelling and grammar help is allowed. Code may be LLM-assisted (see the disclosure rule under Conventions).
- Randomness: create one `np.random.Generator` once at the top and pass it into functions and classes. Never create a new one inside them. Spawn independent train, validation and test streams with `np.random.SeedSequence(2026).spawn(3)`.
- Verify before reporting results: test the simulator and the matcher with known answers. Examples: the depth histogram against the analytic curve, Poisson variance close to the mean, the Compton-edge formula.

## Running

- Notebook: open `spectrum_generator.ipynb` with the working directory set to `proj3/`; it imports from the `.py` modules here (currently still `from spectrum_generator import ...`, which needs updating to `data_reader`).
- Module CLI: `python data_reader.py [path/to/xcom.html]` prints a summary of the parsed arrays.

## Architecture (current state)

- `data_reader.py` (formerly `spectrum_generator.py`): data loading only. It also defines every data path (`DATA_DIR`, `REFERENCE_DIR`, `MEASURED_DIR`, `BACKGROUNDS_DIR`, `DATASETS_DIR`, `FIG_DIR`, `DEFAULT_*_PATH`); use these constants rather than hard-coded paths. Parses the saved NIST XCOM page `data/reference/XCOM_ Photon Cross Sections Data for LaBr3.html` (windows-1252 encoded) by collecting `<pre>` blocks and keeping rows whose last 8 fields are numeric. `read_xcom_table()` returns a dict with `"energyKeV"` and one cm²/g array per column (`coherent_scattering_cm2_g`, `incoherent_scattering_cm2_g` (= Compton), `photoelectric_cm2_g`, `pair_nuclear_cm2_g`, `pair_electron_cm2_g`, `total_with_coherent_cm2_g`, `total_without_coherent_cm2_g`). Energy is converted from MeV to keV, and absorption-edge and duplicate-energy rows are kept (11 duplicates, all below 40 keV).
- `detector.py`: `LaBr3Detector` holds the geometry, density and XCOM dict. `interpolatedLinearAttinuation(keVEnergy, process)` interpolates one XCOM column (by its key) in log-log space and returns 1/cm. It also has the FWHM and efficiency curves and the first-interaction sampler.
- `spectrum_parts.py`: pure functions for spectrum components (e.g. the Klein–Nishina Compton continuum).
- `background_estimation.py`: `snipBackground(counts, maxHalfWindow, useLogLogSqrt=False)`, the SNIP algorithm. It is identical to ROOT `TSpectrum::Background` for a fixed window, and `maxHalfWindow` may also be a per-channel array.
- `extract_backgrounds.py`: reads the `energy_cal_labr` TH2 (30 detectors × 16384 bins of 1 keV) from an OSCAR .root file in `data/measured/`, smooths it (σ = 5 bins), and runs SNIP with half-window 1.5 × FWHM(E). It writes `data/backgrounds/backgrounds_<file>.npz` (check plots go to `figures/`) with `keVBinCentres`, `counts`, `background` and the settings. The real spectra are what the parametric background family should be fitted to. It reads the file with uproot, so no ROOT installation is needed. Only the optional `--compare-root` check, against `TSpectrum::Background`, needs PyROOT. For that, `thisroot.sh` fails in Claude's shell, so export `ROOTSYS=$HOME/root`, `PATH`, `LD_LIBRARY_PATH` and `PYTHONPATH=$HOME/root/lib` directly.
- `photon_transport_mc.py`: a toy photon-transport Monte Carlo for one bare crystal. It tracks photoelectric, Compton and pair production events, including both 511 keV photons. It gives SE/FE, DE/FE, peak-to-total and full-energy efficiency; these set the centre values for the randomised escape-peak intensities. Run it with `--check` to verify the Klein–Nishina sampler. A plain run (about 1 M photons × 113 energies from 100 keV to 20 MeV, a few minutes) writes `data/reference/escape_ratios_LaBr3.csv`. Load that with `read_escape_ratios()` from `data_reader.py` and interpolate with `np.interp`. Re-run it if the geometry or source distance in `detector.py` changes; the CSV header records the settings used.
  - XCOM column map (verified against the HTML header): 0 energy (MeV), 1 coherent, 2 incoherent (Compton), 3 photoelectric, 4 pair (nuclear field), 5 pair (electron field), 6 total with coherent, 7 total without coherent. Mass attenuation coefficients are in cm²/g.
- `spectrum_generator.ipynb`: all the physics lives here (work in progress).
  - `interpolatedLinearAttinuation` converts mass attenuation (cm²/g) times density into linear attenuation (1/cm). It currently uses linear `np.interp`; a log-log variant is commented out.
  - Pair-production curves are set to zero below their thresholds (1022 keV nuclear, 2044 keV electron/triplet) after interpolation on a `geomspace` grid.
  - The `detectorResponse` class contains:
    - Resolution curves (`resolutionCurveEleni`, `resolutionCurveOSCAR`).
    - Efficiency curves (`efficiencyCurveEleniPS/VS/AS`, from the ResearchGate LaBr₃ calibration paper linked in the notebook).
    - `comptonEdge`.
    - `probabilityFirstPairInteraction`, which still uses the 4V/S chord.
    - `probabilityPairAbsorption`, the point-source first-interaction sampler from handoff §6, renamed to camelCase.
    - Single- and double-escape stubs.
- `simulate.py`: skeleton (Level 3) for the spectrum simulator. The flow is `drawSpectrumParameters` (the only randomness) → `peakResponse` per full-energy peak → `expectedSpectrum` (sum, background, threshold, rebin, labels) → `simulateSpectrum` (one Poisson draw) → `makeDataset`/`saveDataset` to `data/datasets/`. Labels are peak positions in fractional channels plus type/area/FWHM, padded with NaN. The user writes the function bodies; `DEFAULT_SETTINGS` holds placeholder ranges.
- `compare_simulation.py [dataset|all] [--detector N]`: overlays a simulated spectrum on a measured OSCAR spectrum from `data/measured/` (28Si `energy_cal_labr`, 1 keV bins; 27Al/106Pd `h_LaBr3_energy`, 2 keV bins, 32 detectors, detector 4 empty). The target is read from the file name and selects the peak list in `TARGETS`. Peak areas and the background level come from one weighted NNLS fit with the 28Si background shape (no background is extracted from the other datasets). Saves `figures/compare_simulation_<target>_detector<N>.png`.
- Background model: `fitBackgroundCoefficients` fits a cubic in ln B from 500 to 11000 keV. Below 500 keV, `lowEnergyBackgroundRatios` provides each 28Si detector's measured shape relative to 500 keV, including the hump at about 150-250 keV (backscatter, multiple scattering, La-138 beta). `backgroundShape(..., lowEnergyRatio, keVLowGrid, humpScale)` joins them continuously. `simulate.py` draws `humpScale` in `humpScaleRange` (0.5-2). The hump is never labelled.
- Folder layout: code in `proj3/`; `figures/` for plots; `data/reference/` (XCOM, escape table), `data/measured/` (.root, git-ignored), `data/backgrounds/` (SNIP .npz), `data/datasets/` (simulated, git-ignored).

## ML design (summary of the handoff)

- A plain regressor does not fit, because the number of peaks varies and peaks have no order. The problem is instead framed as **per-channel (or per-window) classification**, "is a peak centered here?", like 1D segmentation.
- Models: logistic regression on channel windows (baseline), a tree ensemble (random forest or gradient boosting), and a small fully convolutional 1D CNN (the main model). Extra baseline: `scipy.signal.find_peaks`.
- Metrics: precision, recall, F1 and precision-recall curves. **Don't report accuracy**, because the classes are extremely imbalanced. A prediction matches a true peak if it lies within a tolerance scaled by the peak width; fix this rule before seeing results. All test numbers get error bars.
- Data: train and tune on synthetic data only. Split the few real labeled spectra into a tuning group and a final group, and evaluate the final group only once. Unlabeled real spectra are only for visual sanity checks, such as whether the network fires on Compton edges.
- Against distribution shift, **randomize rather than imitate**: use wide ranges for background, widths, peak counts, total counts and amplitudes. Contaminant peaks (C, O, intrinsic activity) count as ordinary peaks.

## Simulator pipeline (planned)

Simulate in keV, then convert to channels, then sample:

1. **Full-energy peaks (Gaussian).** Sigma comes from the OSCAR FWHM curve. Area is a random intensity times the efficiency ε(E). Some peaks get a random width multiplier.
2. **Compton step per peak.** Each full-energy peak gets a step at E_C = E·(2E/511)/(1+2E/511), smeared by the resolution. The summed steps form the continuum.
3. **Escape peaks** for full-energy peaks above 1022 keV. Single escape is at E − 511 keV and double escape at E − 1022 keV. Their widths come from the FWHM at the deposited energy, and their intensities are log-uniform random fractions of the parent.
4. **Other components:**
   - Intrinsic ¹³⁸La peaks at 789 and 1436 keV, plus their beta continuum.
   - An external 511 keV peak.
   - A smooth random background.
   - A low-energy threshold.
5. **Channel mapping:** a random calibration (gain, offset, optionally a small quadratic term) maps energies to channels.
6. **Poisson sampling, once,** on the total expected counts. Draw the total counts over a wide range to act as a difficulty knob.
7. **Labels:** record the true peak positions in channels. Open questions: whether the label is the center channel only or a window around it, and how to label overlapping peaks.

**Noise rule:** broaden and scale before the Poisson draw, never after. Never add ad hoc Gaussian noise.

## Physics conventions

- Use **total attenuation without coherent scattering (XCOM column 7)** for where the photon interacts. Choose the process from the ratios μ_photo/μ_tot, μ_Compton/μ_tot and μ_pair/μ_tot.
- Escape peaks need two μ values: one at the parent energy (depth of the first interaction) and one at 511 keV for the escape step. The 511 keV value should be closer to a full-absorption coefficient than to the total.
- Interpolate XCOM in log-log space. `np.interp` needs increasing x, so handle the duplicated edge energies.
- Geometry: on-axis point source D = 22 cm in front of the face. Use the sampler, not the mean chord 4V/S, which is only valid for isotropic radiation. Photons entering at ρ₀ > R/(1 + ℓ/D) (about 2.3 cm for the 3.5" crystal) leave through the side wall. That is about 73% of the face area, independent of R. The mean path is about 10.6 cm, and P_int is about 0.71 at μ = 0.2 /cm.
- `fractionInteracting` returned by the sampler is a **per-event weight**. Any averaged quantity, such as an escape fraction, must be weighted by it.
- Known limitations to state in the report:
  - Only the first interaction is modelled.
  - Electrons and positrons deposit all their energy (no bremsstrahlung escape).
  - Not modelled: crosstalk, backscatter, coincidence summing, Doppler broadening (beyond random widths), and housing or neighbouring detectors.

## Scope

- Core: the simple simulator, a matcher with tests, the baselines, 2–3 models with hyperparameter searches, a hand-labeled real test set, and the report.
- Optional or related work only, not to build on: Geant4, the `spectrum-builder` package, `curve_fit` peak refinement (Gaussian, double or skewed), and per-detector analysis.
- Open questions:
  - Has the staff approved the project?
  - How many real labeled spectra are there, and from how many detectors?
  - What is the exact channel count? The .root file has the raw spectra as `energy_labr` with 65536 channels × 30 detectors, and calibrated ones as `energy_cal_labr` with 16384 × 1 keV bins. Which binning to give the network is still to decide.
  - Is the efficiency curve full-energy-peak efficiency or total efficiency?
  - Is there an existing OSCAR Geant4 model at OCL?
  - How should the labels be designed?
  - The literature search on ML peak finding is still to do.

## Conventions

- Variable names carry units as camelCase prefixes or suffixes (`keVEnergies`, `cmRadius`, `g_cm3Density`, `*_cm2_g`). The user deliberately renamed the handoff's snake_case names to this style, so keep it. The unit in a name must match the quantity: linear attenuation is 1/cm, mass attenuation is cm²/g. Energies are in keV throughout.
- Terminology: say **peak**, not "line", for gamma energies in names, docstrings and text (e.g. `keVPeakEnergy`). Where it could be ambiguous, say "full-energy peak" versus "single-/double-escape peak".
- Physics assumptions are documented in docstrings (`Assumptions:` / `Limitations:`) with literature references (Knoll, Leo). Add references when introducing new formulas.
- **Generative-AI disclosure is a course requirement.** Any function written with LLM help at **Level 2 or higher** must get an `LLM usage: Level N (Label)` line in its docstring; Level 0–1 (None, Debugging) need no docstring tag. The levels are 0–4: None, Debugging, Snippet, Skeleton, Substantial. `UsesOfGenerativeAI.md` must get a matching table row per file or notebook describing what was done. Update both whenever you contribute code. The report needs an LLM declaration after the conclusions.
