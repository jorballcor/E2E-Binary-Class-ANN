# E2E Binary Classification with Artificial Neural Networks

Welcome to the **E2E Binary Classification with ANN** project!  
This repository showcases an end-to-end solution for binary classification using Artificial Neural Networks (ANNs).  
Dive in to explore how we preprocess data, build models, and evaluate performance in a streamlined workflow.

## 📁 Repository Structure

- **`POC-classification/`**: Contains proof-of-concept notebooks and scripts for binary classification tasks.
- **`POC-regression/`**: Includes proof-of-concept materials for regression analysis.
- **`data/`**: Directory for datasets used in the project.
- **`logs/`**: Stores training logs and model performance metrics.
- **`utils/`**: Utility scripts and helper functions to support the main codebase.
- **`app_exiting_classif.py`**: Main application script for executing the binary classification pipeline.
- **`app_salary_regression.py`**: Application script dedicated to salary prediction using regression techniques.
- **`requirements.txt`**: Lists all Python dependencies required to run the project.

## 🚀 Getting Started

To get up and running with this project, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/jorballcor/E2E-Binary-Class-ANN.git
cd E2E-Binary-Class-ANN
```

### 2. Set Up the Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```


### 3. Install Dependencies

```bash
pip install -r requirements.txt
```
This will do the trick, but if you want to feel *REAL* speed, you should check [*uv documentation*](https://docs.astral.sh/uv/) for dependencies management.

### 4. Run the Application

This project uses Streamlit for local and web deployment. Streamlit is the main library for launching the applications.

#### To run the apps locally, use the following commands:
For binary classification:

```bash
streamlit run app_exiting_classif.py
```

For salary regression:
```bash
streamlit run app_salary_regression.py
```

#### Online Deployment
Both applications are also available online. Here are the links:

- [Churn prediction app](https://e2e-binary-class-ann-ku6mnpgbqxjbcuprxqje8m.streamlit.app/)
- [Salary prediction app](https://e2e-binary-class-ann-kfzui3bk4p8blemug22dkh.streamlit.app/)

Visit [my streamlit profile](https://share.streamlit.io/user/jorballcor "Jorge Ballester Streamlit profile") for more content like this.
 

## 🧠 Project Overview
This project demonstrates the application of Artificial Neural Networks in binary classification and regression scenarios. By following the provided scripts and notebooks, you'll learn how to:

- Preprocess and clean datasets.
- Build and train ANN models.
- Evaluate model performance using various metrics.
- Deploy models locally and online for practical use cases.


## 🤝 Contributing
We welcome contributions! If you'd like to improve this project, please fork the repository and create a pull request. For major changes, open an issue first to discuss what you'd like to modify.

## 📄 License
This project is licensed under the MIT License. 

## 📞 Contact
For questions or feedback, please reach out to [jorballcor](https://www.linkedin.com/in/jorballcor/ "Jorge Ballester LinkedIn Homepage")
.
