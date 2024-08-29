# FDA
This repository shows how combining different models, trained on various levels of data, can lead to better predictions.

## Data
The **train.csv.zip** file contains the data used for comparing different methods. This dataset is from a Kaggle competition and comes from Rossmann, a European pharmacy chain in seven countries. You can also access the dataset [here](https://www.kaggle.com/competitions/rossmann-store-sales).

## Code
1. **data_propressing.ipynb**: This Jupyter Notebook demonstrates how the data is processed.
2. **all_methods.py**:  This Python file contains all the algorithm functions used by the main program. It includes our FDA linear and FDA tree methods, along with other benchmark algorithms like Decoupled OLS, Shared OLS, DAC, Shrunken SAA, PAB linear, and PAB tree.
3. **main.ipynb**: This is the main Jupyter Notebook that runs the different algorithms, calculates costs, and tracks runtime.
