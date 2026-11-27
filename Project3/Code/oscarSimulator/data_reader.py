"""Extract the photon attenuation table from a saved NIST XCOM HTML page.

LLM usage: Level 4 (Substantial); an LLM drafted most of this script.
"""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path

import numpy as np


# Folder layout (all paths relative to this file, so scripts work from any working directory)
PROJECT_DIR = Path(__file__).parent
FIG_DIR = PROJECT_DIR / "figures"
DATA_DIR = PROJECT_DIR / "data"
REFERENCE_DIR = DATA_DIR / "reference"      # physics inputs: XCOM table, escape-ratio table
MEASURED_DIR = DATA_DIR / "measured"        # OSCAR .root files
BACKGROUNDS_DIR = DATA_DIR / "backgrounds"  # SNIP backgrounds from extract_backgrounds.py
DATASETS_DIR = DATA_DIR / "datasets"        # simulated spectra from simulate.py

DEFAULT_HTML_PATH = REFERENCE_DIR / "XCOM_ Photon Cross Sections Data for LaBr3.html"
DEFAULT_ESCAPE_RATIOS_PATH = REFERENCE_DIR / "escape_ratios_LaBr3.csv"
DEFAULT_BACKGROUND_PATH = BACKGROUNDS_DIR / "backgrounds_si28-sirius-20190313-142346.npz"



class _PreformattedTextParser(HTMLParser):
    """Collect the text contents of HTML ``<pre>`` blocks.

    LLM usage: Level 4 (Substantial); an LLM drafted most of this class.
    """

    def __init__(self) -> None:
        """Initialize storage for completed and currently open blocks.

        LLM usage: Level 4 (Substantial).
        """
        super().__init__()
        self.pre_blocks: list[str] = []
        self._current_block: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Begin collecting text when an opening ``<pre>`` tag is encountered.

        LLM usage: Level 4 (Substantial).
        """
        if tag.lower() == "pre":
            self._current_block = []

    def handle_data(self, data: str) -> None:
        """Append HTML text to the currently open ``<pre>`` block.

        LLM usage: Level 4 (Substantial).
        """
        if self._current_block is not None:
            self._current_block.append(data)

    def handle_endtag(self, tag: str) -> None:
        """Save a collected block when its closing ``</pre>`` tag is reached.

        LLM usage: Level 4 (Substantial).
        """
        if tag.lower() == "pre" and self._current_block is not None:
            self.pre_blocks.append("".join(self._current_block))
            self._current_block = None


def read_xcom_table(
    html_path: str | Path = DEFAULT_HTML_PATH,
) -> dict[str, np.ndarray]:
    """Parse the XCOM table into a dict of NumPy arrays, one per column.

    "energyKeV" is the photon energy in keV. All other entries are mass
    attenuation coefficients in cm²/g: coherent and incoherent (Compton)
    scattering, photoelectric absorption, pair production in the nuclear and
    electron fields, and total attenuation with and without coherent scattering.
    Absorption-edge rows and repeated energies are retained.

    Example:
        xcomTables = read_xcom_table()
        totalWithoutCoherent = xcomTables["total_without_coherent_cm2_g"]

    LLM usage: Level 4 (Substantial); an LLM drafted most of this function.
    """
    path = Path(html_path)
    html = path.read_text(encoding="windows-1252")

    parser = _PreformattedTextParser()
    parser.feed(html)

    rows: list[list[float]] = []
    for block in parser.pre_blocks:
        for line in block.splitlines():
            fields = line.split()
            if len(fields) < 8:
                continue
            try:
                # The final eight fields are the numerical XCOM table columns.
                row = [float(value) for value in fields[-8:]]
            except ValueError:
                continue
            rows.append(row)

    if not rows:
        raise ValueError(f"No XCOM cross-section rows found in {path}")

    data = np.asarray(rows, dtype=float)
    xcomTabels  = {
        "energyKeV":  data[:, 0]*1e3,  # keV
        "coherent_scattering_cm2_g": data[:, 1], 
        "incoherent_scattering_cm2_g": data[:, 2],
        "photoelectric_cm2_g": data[:, 3],
        "pair_nuclear_cm2_g": data[:, 4],
        "pair_electron_cm2_g": data[:, 5],
        "total_with_coherent_cm2_g": data[:, 6],
        "total_without_coherent_cm2_g": data[:, 7],
    }
    return xcomTabels


def read_escape_ratios(
    csv_path: str | Path = DEFAULT_ESCAPE_RATIOS_PATH,
) -> dict[str, np.ndarray]:
    """Read the escape-peak table written by photon_transport_mc.py.

    Returns a dict with one array per CSV column, e.g. "energy_keV",
    "se_over_fe", "de_over_fe", "fe_over_interacting" (see the '#' lines at the
    top of the CSV for definitions and the simulation settings). Rows are sorted
    by energy, so values for any full-energy peak energy can be interpolated with np.interp:

    Example:
        escapeRatios = read_escape_ratios()
        singleEscapeOverFullEnergy = np.interp(
            4438.0, escapeRatios["energy_keV"], escapeRatios["se_over_fe"]
        )

    LLM usage: Level 4 (Substantial); an LLM wrote this function.
    """
    lines = [
        line for line in Path(csv_path).read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    columnNames = lines[0].split(",")
    values = np.loadtxt(lines[1:], delimiter=",", ndmin=2)
    return {name: values[:, column] for column, name in enumerate(columnNames)}


def main() -> None:
    """Parse the optional input path and print a summary of the XCOM table.

    LLM usage: Level 4 (Substantial).
    """
    argument_parser = argparse.ArgumentParser(
        description="Extract the photon attenuation table from an XCOM HTML file."
    )
    argument_parser.add_argument(
        "html_path",
        nargs="?",
        type=Path,
        default=DEFAULT_HTML_PATH,
        help="XCOM HTML file (defaults to the LaBr3 file beside this script)",
    )
    args = argument_parser.parse_args()

    xcomTables = read_xcom_table(args.html_path)
    energyKeV = xcomTables["energyKeV"]
    print(f"Read {energyKeV.size} energy points.")
    print(f"Energy range: {energyKeV[0]:g}-{energyKeV[-1]:g} keV")
    for key, values in xcomTables.items():
        if key != "energyKeV":
            print(f"{key}: shape {values.shape}, range {values.min():g}-{values.max():g} cm²/g")


if __name__ == "__main__":
    main()

