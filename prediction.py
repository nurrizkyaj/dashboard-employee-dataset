import pandas as pd
import pickle

# Function to load a trained model
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Function to make predictions using the loaded model
def prediction(model, data):
    predictions = model.predict(data)
    return predictions

def prepare_employee_data():
    """
    Retrieve new employee data entered by the user in DataFrame format.

    Returns:
        pd.DataFrame: Employee data as a DataFrame.
    """
    data = pd.DataFrame({
        'Age': [25],
        'BusinessTravel': [1],
        'DailyRate': [800],
        'Department': [2],
        'DistanceFromHome': [2],
        'Education': [3],
        'EducationField': [2],
        'EnvironmentSatisfaction': [3],
        'Gender': [1],
        'HourlyRate': [60],
        'JobInvolvement': [3],
        'JobLevel': [1],
        'JobRole': [2],
        'JobSatisfaction': [4],
        'MaritalStatus': [2],
        'MonthlyIncome': [6000],
        'MonthlyRate': [12000],
        'NumCompaniesWorked': [1],
        'OverTime': [1],
        'PercentSalaryHike': [15],
        'PerformanceRating': [3],
        'RelationshipSatisfaction': [2],
        'StockOptionLevel': [0],
        'TotalWorkingYears': [7],
        'TrainingTimesLastYear': [2],
        'WorkLifeBalance': [3],
        'YearsAtCompany': [2],
        'YearsInCurrentRole': [3],
        'YearsSinceLastPromotion': [1],
        'YearsWithCurrManager': [2]
    })
    return data

def display_prediction_results(rf_result, xgb_result):
    """
    Display prediction results from two different models.

    Args:
        rf_result (int): Prediction from the Random Forest model.
        xgb_result (int): Prediction from the XGBoost model.
    """
    print("\n=====================================================================================")
    print("Attrition Prediction:")
    print(f"Random Forest (RF) Prediction: {'Attrition' if rf_result[0] == 1 else 'No Attrition'}")
    print(f"XGBoost (XGB) Prediction: {'Attrition' if xgb_result[0] == 1 else 'No Attrition'}")
    print("=====================================================================================\n")
    
def main():
    """
    Main function to gather input data from user and generate predictions using two models.
    """
    # Prepare new employee data
    employee_data = prepare_employee_data()

    # Features required according to the trained models
    required_features = [
        'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome', 'Education', 
        'EducationField', 'EnvironmentSatisfaction', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime', 
        'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'YearsSinceLastPromotion', 
        'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 
        'YearsWithCurrManager', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 
        'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating'
    ]
    
    # Adjust data to fit the features required by the models
    employee_data = employee_data[required_features].values

    # Load the trained models
    rf_model_path = './models/rf_model.pkl'  
    xgb_model_path = './models/xgb_model.pkl'  

    # Load Random Forest and XGBoost models
    rf_model = load_model(rf_model_path)
    xgb_model = load_model(xgb_model_path)
    
    # Make predictions with both models
    rf_predictions = prediction(rf_model, employee_data)
    xgb_predictions = prediction(xgb_model, employee_data)
    
    # Display prediction results
    display_prediction_results(rf_predictions, xgb_predictions)

if __name__ == "__main__":
    main()