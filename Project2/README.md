<div style="background-color:#F0DAE2; color:#823C55; border-radius:12px; padding:16px; width:90%;">

## FYS-STK4155 **Project 2**

The legendary ponies: Artis Birze & Maia Hermione Oppedal Starr
</div>

This project implements a feed-forward neural network (NN) from scratch, trained
with backpropagation, and uses it on two problems:

- **Regression** on noisy samples of the Runge function, where the NN is compared to
  OLS, Ridge and Lasso regression (parts B to D).
- **Classification** of handwritten digits from MNIST, where the NN is compared to
  logistic regression and to scikit-learn's `MLPClassifier` (part E)

We study the choices that matter when training the network: optimizer and
learning rate, architecture (activation, depth, width), batch size and L1/L2
regularization. Hyperparameters are selected with k-fold cross-validation 
(regression) or a validation split (MNIST), and the differences between the
models are compared against their standard errors. Part F is a critical
evaluation of the methods, with tables of accuracy, training time and tuning
effort. It also includes a gradient check of our backpropagation against JAX.

---

This project repository has the following structure:

| Folder | File | Description |
| --- | --- | --- |
| `Code` | `Project_2.ipynb` | For running all code to create figures and results, as per the Project 2 demands |
| -- | `config.py` | Setting default paths, seed and number of CV folds |
| -- | `utils.py` | Runge-function data, polynomial features and OLS/Ridge/Lasso helpers (reused from Project 1), k-fold CV, caching of results |
| -- | `plot_params.py` | Matplotlib settings matching our LaTeX column width |
| `Code/Data` | `data.py` | Functions for loading, splitting and one-hot encoding data (MNIST) |
| -- | `plots.py` | Shared plotting helpers: error bands, heatmap markings, log ticks |
| `Code/Machine` | `activations.py` | Activation functions and their derivative forms |
| -- | `costs.py` | Cost functions and their derivative forms |
| -- | `metrics.py` | Functions for analyzing results (`mse_score`, `accuracy`, `confusion_matrix`, precision, recall, and F1, standard erros, and paired comparisons) |
| -- | `network.py` | `NeuralNetwork` (NN) class which calls functions from all other *Machine* files, and scikit-learn style wrappers (`NeuralNetRegressor`, `NeuralNetClassifier`, `make_nn`) for use with cross-validation |
| -- | `optimizers.py` | Gradient descent variants (plain, momentum, Adagrad, RMSProp, Adam) and the training loop |

The `Project_2.ipynb` file is the main operating point, where parts B to F from Project 2 are done. Cells are briefly explained in markdown cells above, and are
divided into sections based on their parts.

All results are stored in the *Results* folder, as to not have to rerun all heavy
computation repetitively. The figures are also stored in the *Figures* folder, saved
as PDFs with names mirroring the tasks they're a part of. Result fines for the
final MNIST model have their settings in the file name, so changing a setting reruns 
it instead of loading an outdated file. Set `RERUN = True` in `Project_2.ipynb` to
recompute everything.

The MNIST data is located at `Code/Data/mnist.npz`.

---

Installing all the required packages by running the command
```
pip3 install -r requirements.txt
```
from the project folder.

