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
