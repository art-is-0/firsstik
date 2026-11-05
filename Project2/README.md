<div style="background-color:#F0DAE2; color:#823C55; border-radius:12px; padding:16px; width:90%;">

## FYS-STK4155 **Project 2**

The legendary ponies: Artis Birze & Maia Hermione Oppedal Starr
</div>

*Short description of the project*

---

This project repository has the following structure (in the `Code` folder):

| Folder | File | Description |
| --- | --- | --- |
|  | `config.py` | Setting default paths and seed |
|  | `main.ipynb` | For running all code to create figures and results, as per the Project 2 demands |
|  | `utils.py` | Reused function-file from Project 1 |
| Data | `data.py` | Functions for loading and adjusting data |
| Machine | `activations.py` | Activation functions and their derivative forms |
| -- | `costs.py` | Cost functions and their derivative forms |
| -- | `metrics.py` | Functions for analyzing results (`mse_score`, `accuracy`, `confusion_matrix`, precision, recall, and F1) |
| -- | `network.py` | `NeuralNetwork` (NN) class which calls functions from all other *Machine* files. |
| -- | `optimizers.py` | Functions for optimizing the NN. |

The `main.ipynb` file is the main operating point, where parts B to F from Project 2
are done.  Each cell has part and purpose listed at the top, divided into sections.

All results are stored in the *Results* folder, as to not have to rerun all heavy
computation repetitively. The figures are also stored in the *Figures* folder, saved
as PDFs with names mirroring the tasks they're a part of.

---

Installing all the required packages by running the command
```
pip3 install -r requirements.txt
```
from the project folder.

