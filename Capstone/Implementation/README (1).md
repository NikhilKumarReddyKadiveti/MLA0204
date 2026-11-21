# EduPredict – Student Performance Analytics

**Project:** Student Performance Prediction Using Decision Tree Algorithm  
**Algorithm:** Decision Tree Classification using Entropy / Information Gain  
**Interface:** Streamlit  
**Dataset:** Synthetic student records (300 rows)

## 1. Project overview

EduPredict is a Machine Learning application that predicts a student's academic performance category:

- Low
- Medium
- High

The model uses attendance, previous exam marks, assignment marks, internal marks, study hours, class participation, and assignment completion.

The application includes a dashboard, individual predictions, dataset management, model training, model evaluation, decision-tree visualization, batch prediction, and project information.

## 2. Project files

Keep these files in the same project folder:

```text
EduPredict/
├── app.py
├── generate_dataset.py
├── requirements.txt
├── sample_students.csv
└── README.md
```

The application imports `generate_students` from `generate_dataset.py`, so do not rename that file or move it to another folder.

## 3. Requirements

- Windows 10 or Windows 11
- Python 3.10 or newer recommended
- Internet connection for the first installation of Python packages
- VS Code or another Python editor (optional)

## 4. Installation and execution on Windows

Open a terminal in the EduPredict project folder and run:

### Create a virtual environment

```bash
python -m venv venv
```

### Activate it

```bash
venv\Scripts\activate
```

### Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Generate or regenerate the sample dataset

```bash
python generate_dataset.py
```

The generator creates `sample_students.csv` in the project folder. It also creates `.streamlit/config.toml` if that theme configuration does not already exist.

### Start the application

```bash
streamlit run app.py
```

Open the local address shown in the terminal (normally `http://localhost:8501`) if the browser does not open automatically.

To stop the application, press `Ctrl+C` in the terminal.

## 5. Dataset format

The application expects the following columns:

| Column | Meaning | Expected values |
|---|---|---|
| `Student_ID` | Student identifier | Text (optional; generated if absent) |
| `Attendance` | Attendance percentage | 0–100 |
| `Previous_Exam_Marks` | Previous examination marks | 0–100 |
| `Assignment_Marks` | Assignment marks | 0–100 |
| `Internal_Marks` | Internal assessment marks | 0–100 |
| `Study_Hours` | Study hours per day | 0–24 |
| `Class_Participation` | Participation level | Low, Medium, High |
| `Assignment_Completion` | Assignment completion percentage | 0–100 |
| `Performance` | Target class | Low, Medium, High |

For model training, the CSV must include all seven input features and the `Performance` target column. `Student_ID` is optional.

For batch prediction, provide the seven input features. The `Performance` target column is not required.

## 6. Model and evaluation

The application uses Scikit-learn's `DecisionTreeClassifier` with:

- `criterion="entropy"`
- An 80:20 train-test split
- Stratification when the dataset has enough examples in each class
- A fixed random state for repeatable splitting
- Median imputation for missing numeric values
- Most-frequent imputation and ordinal encoding for class participation

The preprocessing pipeline is fitted on the training split to reduce data leakage. The application reports accuracy, weighted precision, weighted recall, weighted F1-score, a confusion matrix, and feature importance using the held-out test data.

## 7. Synthetic-data notice

`sample_students.csv` contains randomly generated synthetic records. It does not represent real students.

The sample target labels are generated using a weighted score based on the input features and random variation. Therefore, the model learns patterns created by that synthetic labeling process. Results from this dataset are for software testing and academic demonstration only; they do not establish how accurately the model predicts real student outcomes.

For a meaningful real-world evaluation, obtain an appropriately authorized academic dataset, check its quality and suitability, and evaluate the model on data that was not used to create or tune it.

## 8. Common issues

**`ModuleNotFoundError`**  
Activate the virtual environment and run `pip install -r requirements.txt` again.

**`streamlit` is not recognized**  
Use `python -m streamlit run app.py`.

**`No module named generate_dataset`**  
Make sure `generate_dataset.py` is in the same folder as `app.py`.

**Dataset missing-column error**  
Check that the CSV headers match the column names in the dataset table above.

**Port 8501 is already in use**  
Run `streamlit run app.py --server.port 8502`.

## 9. Academic use

Use this application as a decision-support demonstration, not as the sole basis for decisions about students. Verify predictions with educators and appropriate real-world evidence.
