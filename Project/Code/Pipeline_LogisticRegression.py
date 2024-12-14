import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    classification_report
)

# 1. Load the dataset
def load_data(file_path):

    df = pd.read_csv(file_path)
    
    # Verify column names and data
    print(f"Dataset from {file_path}")
    print("Dataset shape:", df.shape)
    print("\nColumn names:", df.columns)
    print("\nLabel distribution:")

    if 'label_sexist' in df.columns:
        print(df['label_sexist'].value_counts(normalize=True))
        return df['text'], df['label_sexist'], df['rewire_id']
    else: 
        print('There is no sexist_label')
        return df['text'], df['rewire_id']
    
    

# 2. Vectorize text data using TF-IDF
def vectorize_text(X_train, X_val, X_test):

    # Create TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,  # Limit to top 5000 features
        stop_words='english'  # Remove English stop words
    )
    
    # Fit and transform training data
    X_train_vectorized = vectorizer.fit_transform(X_train)
    
    # Transform validation and test data
    X_val_vectorized = vectorizer.transform(X_val)
    X_test_vectorized = vectorizer.transform(X_test)
    
    return X_train_vectorized, X_val_vectorized, X_test_vectorized, vectorizer

# 3. Train Logistic Regression model
def train_logistic_regression(X_train_vectorized, y_train, X_val_vectorized, y_val):
    # pipeline for parameter tuning
    pipeline = Pipeline([
        #('tfidf', TfidfVectorizer()),  
        ('clf', LogisticRegression(random_state=42))
    ])
    # Define parameter grid for grid search
    param_grid = {
        #'tfidf__max_features': [1000, 2000, 5000],
        #'tfidf__ngram_range': [(1,1), (1,2)],
        'clf__C': [0.001, 0.01, 0.1, 1, 10, 20],
        'clf__penalty': ['l2', 'l1'],
        'clf__solver': ['liblinear'],
        'clf__max_iter': [100, 200, 500, 1000, 1100],
        'clf__class_weight': [
        {0: 0.5, 1: 2.0},
        {0: 1.0, 1: 1.0},  # Equal weights
        {0: 0.5, 1: 3.0},
        {0: 0.2, 1: 3.0},
        {0: 1, 1: 4.0}
    ]
    }
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        scoring='f1_weighted',  
        n_jobs=-1  
    )
    custom_weights = {0: 0.5, 1: 2.0} 
    # Initialize and train the model
    model = LogisticRegression(
        #penalty = "l1",
        #C=11,
        #solver= "liblinear",
        max_iter=1000,#100,  
        class_weight='balanced' #custom_weights#'balanced'  
    )
    # the following code line can be used for the RandomForestClassifier
    #model = RandomForestClassifier(n_estimators = 10, criterion='entropy', random_state = 0)
    #grid_search.fit(X_train_vectorized, y_train)
    model.fit(X_train_vectorized, y_train)
    
    # Validate model performance
    
    #print("Best parameters:", grid_search.best_params_)
    #print("Best cross-validation score:", grid_search.best_score_)
    val_pred = model.predict(X_val_vectorized)
    print("\nValidation Performance:")
    print(classification_report(y_val, val_pred))
    
    return model

# 4. Evaluate model performance on test set
def evaluate_model(model,X_test, X_test_vectorized, y_test_path, vectorizer, test_ids):

    # Make predictions
    df_test = pd.read_csv(y_test_path)
    y_test = df_test['label_sexist']
    y_pred = model.predict(X_test_vectorized)
    y_pred_proba = model.predict_proba(X_test_vectorized)[:, 1]
    
    # Print detailed performance metrics
    print("\nTest Set Performance:")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Calculate additional metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\nMetrics Summary:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Create detailed predictions DataFrame
    predictions_df = pd.DataFrame({
        'rewire_id': test_ids,
        'text': X_test,
        'tokenized_text': vectorizer.inverse_transform(X_test_vectorized),
        'true_label': y_test,
        'predicted_label': y_pred,
        'prediction_probability': y_pred_proba
    })
    
    # Save predictions to CSV
    predictions_df.to_csv('output/logisticRegression_test_predictions.csv', index=False)
    # the following code line can be used for the RandomForestClassifier
    #predictions_df.to_csv('output/RandomForest_test_predictions.csv', index=False)
# Main execution
def main(train_path, val_path, test_path, test_label_path):

    # 1. Load data
    X_train, y_train, train_ids = load_data(train_path)
    X_val, y_val, val_ids = load_data(val_path)

    X_test, test_ids = load_data(test_path)
    
    # 2. Vectorize text
    X_train_vectorized, X_val_vectorized, X_test_vectorized, vectorizer = vectorize_text(
        X_train, X_val, X_test
    )
    
    # 3. Train model
    model = train_logistic_regression(
        X_train_vectorized, y_train, 
        X_val_vectorized, y_val
    )
    
    # 4. Evaluate model on test set
    evaluate_model(model, X_test, X_test_vectorized, test_label_path, vectorizer, test_ids)

# Run the script
if __name__ == '__main__':
    main(
        train_path='data/preprocessed/DF_train.csv',
        val_path='data/preprocessed/DF_dev.csv',
        test_path='data/preprocessed/DF_test.csv',
        test_label_path = 'data/preprocessed/Actual_labels.csv'
    )