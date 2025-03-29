# Particle Physics Analysis using Machine Learning

This repository combines projects that apply machine learning techniques to analyze high-energy physics events, focusing on classifying and predicting jet counts using datasets from the CMS (Compact Muon Solenoid) experiment.

---

## Overview

The primary objectives of this repository are:

1. **Jet Count Prediction**: Predict the number of jets in a particle collision event using physics-inspired features derived from CMS Open Data.
2. **Exploratory Data Analysis**: Perform EDA on the dataset 'Higgs Boson Machine Learning Challenge'.
3. **Event Classification**: Classify particle collision events to distinguish between signal and background events.

By leveraging machine learning models, we aim to explore the feasibility of predicting jet counts and classifying events using kinematic variables.

---

## Repository Structure

- `Data/`: Contains datasets used for analysis.
- `notebooks/`: Jupyter notebooks with detailed implementations of data analysis, model training, and evaluation.
  - `Jet_Prediction_using_ML.ipynb`: Notebook focusing on predicting jet counts.
  - `EDA.ipynb`: Notebook for performing EDA on the Higgs Boson ML Challenge.
  - `EventClassifier.ipynb`: Notebook dedicated to classifying collision events.
- `requirements.txt`: Lists the Python dependencies required to run the notebooks.

---

## Dataset Description

The datasets contain information from particle collisions with key features such as:

- `Run`, `Lumi`, `Event`: Identifiers for each collision event.
- `MR`, `Rsq`: Razor kinematic variables estimating event mass scale and energy flow.
- `E1`, `Px1`, `Py1`, `Pz1`: Four-vector of the **leading megajet**.
- `E2`, `Px2`, `Py2`, `Pz2`: Four-vector of the **subleading megajet**.
- `HT`: Scalar sum of transverse momentum of all jets.
- `MET`: Missing transverse energy.
- `nJets`: Target variable — number of jets with transverse momentum > 40 GeV.
- `nBJets`: Number of **b-tagged** jets.

The other dataset is not added due to privacy reasons. Visit [Kaggle]{https://www.kaggle.com/competitions/higgs-boson} for the dataset.

**Source**:  
Duarte, Javier (2015). 
[CERN Open Data Portal](http://opendata.cern.ch/)

[Higgs Boson Machine Learning Challenge]{https://kaggle.com/competitions/higgs-boson}., 2014. Kaggle.

---

## Technologies Used

- Python
- Jupyter Notebook
- Pandas & NumPy
- Scikit-learn
- Matplotlib & Seaborn
- XGBoost

---

## How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/soumyashaw/Particle-Physics-using-ML.git
   cd Particle-Physics-using-ML
   ```
2.	Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebooks:
	- Open notebooks/Jet_Prediction_using_ML.ipynb in Jupyter Notebook or JupyterLab and execute the cells to follow the jet prediction analysis.
	- Open notebooks/EventClassifier.ipynb to explore the event classification process.

---

## Acknowledgements

Special thanks to:
	- CERN Open Data and Javier Duarte for providing the dataset.
	- The open-source Python data science community for the powerful tools used in this analysis, including Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn.

This work is intended for educational and research purposes only.
