# NLP-Group 5
# **Sexism Detection on Social Media**

## **Task Description**
The goal of this project is to perform **binary classification** of short utterances on social media to determine whether they are **sexist** or **non-sexist**. 

For this task, we used the **EDOS Dataset** ("Explainable Detection of Online Sexism"), specifically focusing on the subset that was **already annotated**. This subset contains approximately **20,000 samples** of short utterances labeled as sexist or non-sexist.

To solve this classification task, we built and compared two models:
1. **Logistic Regression**: A non-deep learning approach serving as a baseline.
2. **RoBERTa**: A state-of-the-art transformer-based deep learning model.

The notebooks in the Code folder contain detailed information about the implementation and the results of the analysis.

---

## **Project Structure**
The project is organized into the following folders:

### **1. Code**
This folder contains the core implementation of the models and related analyses:
- **Jupyter Notebooks** used for preprocessing of the dataset.
- Implementations of the **Logistic Regression** and **RoBERTa** models.
- **Jupyter Notebooks** used for training, evaluation, and in-depth analysis of the models.

### **2. Relevant Documents**
This folder includes additional analyses and findings:
- Text files with detailed observations and conclusions drawn from the project.

### **3. Data**
The **Data** folder contains the input dataset with two subfolders:
- **Raw**: The original EDOS dataset without modifications.
- **Preprocessed**: The cleaned and preprocessed version of the dataset used for model training and evaluation.

### **4. Output**
This folder contains the outputs generated from the models:
- CSV files with predictions and results from the Logisitc Regression model.


---

## **Contributors**
This project was a collaborative effort by the following contributors:
1. **Aditi**
2. **Arash Behaein**
3. **Babak Bayani**
4. **Juliane Leibhammer**

We worked together to design, implement, and analyze the performance of both models to effectively detect sexist content.

---





