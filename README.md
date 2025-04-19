# FDA
This repository shows how combining different models, trained on various levels of data, can lead to better predictions and decisions.

## Data and data processing
1. The **train.csv.zip** file contains the data used for comparing different methods. This dataset is from a Kaggle competition and comes from Rossmann, a European pharmacy chain in seven countries. You can also access the dataset [here](https://www.kaggle.com/competitions/rossmann-store-sales).
2. **data_propressing.ipynb**: This Jupyter Notebook demonstrates how the data is processed.

## Code in FDA_prediction
1. **all_methods.py**:  This Python file contains all the algorithm functions used by the main program. It includes our FDA (linear+linear), FDA (linear + random forest) and FDA (SAA + linear) methods, along with other benchmark algorithms like Decoupled OLS, Shared OLS, DAC, Shrunken SAA, PAB linear, random forest and random forest with product index.
2. **main.ipynb**: This is the main Jupyter Notebook that runs the different algorithms, calculates costs, and tracks runtime.

## Code in FDA_decision
1. **all_methods_newsvendor.py**:  This Python file contains all the algorithm functions used by the main program. It includes our FDA (linear+linear), FDA (linear + random forest) and FDA (SAA + linear) methods, along with other benchmark algorithms like Decoupled KO, Pooled KO, DAC and Shrunken SAA.
2. **main_decision.ipynb**: This is the main Jupyter Notebook that runs the different algorithms, calculates costs.
