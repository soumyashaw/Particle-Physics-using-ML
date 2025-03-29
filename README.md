# Particle-Physics-using-ML

This project focuses on applying machine learning techniques to classify and predict jet counts from high-energy physics events using the **MultiJet dataset** from the CMS (Compact Muon Solenoid) experiment.

---

## 📘 Overview

The goal of this notebook is to predict the number of jets in a particle collision event using key physics-inspired features derived from CMS Open Data. By leveraging machine learning models, we aim to explore the feasibility of jet prediction using kinematic variables.

---

## 📊 Dataset Description

The dataset contains information from particle collisions and includes the following key features:

- `Run`, `Lumi`, `Event`: Identifiers for each collision event.
- `MR`, `Rsq`: Razor kinematic variables estimating event mass scale and energy flow.
- `E1`, `Px1`, `Py1`, `Pz1`: Four-vector of the **leading megajet**.
- `E2`, `Px2`, `Py2`, `Pz2`: Four-vector of the **subleading megajet**.
- `HT`: Scalar sum of transverse momentum of all jets.
- `MET`: Missing transverse energy.
- `nJets`: Target variable — number of jets with transverse momentum > 40 GeV.
- `nBJets`: Number of **b-tagged** jets.

**Source**:  
Duarte, Javier (2015). *Example CSV output file for SUSYBSMAnalysis-RazorFilter.*  
[CERN Open Data Portal](http://opendata.cern.ch/)

---

## 🛠️ Workflow

1. **Data Loading and Cleaning**
2. **Exploratory Data Analysis (EDA)**
3. **Feature Engineering**
4. **Train/Test Split**
5. **Model Training (e.g., Random Forest, XGBoost)**
6. **Model Evaluation**
7. **Prediction and Visualization**

---

## 🚀 Technologies Used

- Python
- Jupyter Notebook
- Pandas & NumPy
- Scikit-learn
- Matplotlib & Seaborn
- XGBoost

---

## 📈 Results

The trained model is evaluated using common metrics like MAE, RMSE, and R² score. Visualizations of prediction accuracy and feature importance are also provided.

---

## 📁 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/jet-prediction-ml.git
   cd jet-prediction-ml
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   Open `Jet Prediction using ML.ipynb` in Jupyter Notebook or JupyterLab and execute the cells.

---

## 📜 License

This project is licensed under the MIT License.

---

## 🤝 Acknowledgements

Thanks to CERN Open Data and Javier Duarte for providing the dataset.







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
   cd Particle-Physics-using-ML```
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
	•	CERN Open Data and Javier Duarte for providing the dataset.
	•	The open-source Python data science community for the powerful tools used in this analysis, including Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn.

This work is intended for educational and research purposes only.
