import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ModelEvaluator:
    """
    A class to handle the evaluation of a machine learning model's performance.
    It calculates various metrics such as R², MSE, RMSE, and MAE.
    """

    @staticmethod
    def evaluate_model(y_true, y_pred, dataset_type):
        """
        Evaluate the model's performance on a given dataset.

        Parameters:
        - y_true: Actual target values.
        - y_pred: Predicted target values.
        - dataset_type: Type of dataset (e.g., 'Training' or 'Testing').
        """
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)

        print(f"{dataset_type} Set Performance:")
        print(f"R² Score: {r2}")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"Mean Absolute Error (MAE): {mae}")
        print()


class RandomForestModel:
    """
    A class for training and evaluating a Random Forest model for regression.
    """

    def __init__(self, data_path):
        """
        Initialize the RandomForestModel with dataset path and load the data.

        Parameters:
        - data_path: Path to the dataset CSV file.
        """
        self.df = pd.read_csv(data_path)
        self.rf_regressor = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )

    def prepare_data(self):
        """
        Prepare features and target variables (X and y).
        """
        X = self.df[
            [
                "Type_of_Property",
                "Number_of_Rooms",
                "Living_Area",
                "Fully_Equipped_Kitchen",
                "Terrace",
                "Garden",
                "Surface_area_plot_of_land",
                "Number_of_Facades",
                "Region_Code",
                "AS_NEW",
                "GOOD",
                "JUST_RENOVATED",
                "TO_BE_DONE_UP",
                "TO_RENOVATE",
                "TO_RESTORE",
                "APARTMENT",
                "DUPLEX",
                "HOUSE",
                "PENTHOUSE",
                "TOWN_HOUSE",
                "VILLA",
            ]
        ]
        y = self.df["Price"]
        return X, y

    def train_model(self, X_train, y_train):
        """
        Train the Random Forest model using the provided training data.

        Parameters:
        - X_train: Feature data for training.
        - y_train: Target data for training.
        """
        self.rf_regressor.fit(X_train, y_train)

    def predict(self, X):
        """
        Make predictions using the trained model.

        Parameters:
        - X: Feature data for prediction.
        """
        return self.rf_regressor.predict(X)

    def cross_validate(self, X, y):
        """
        Perform cross-validation on the model to evaluate its performance.

        Parameters:
        - X: Feature data.
        - y: Target data.
        """
        cv_scores = cross_val_score(
            self.rf_regressor, X, y, cv=10, scoring="r2", n_jobs=-1
        )
        print("Cross-Validated R² Scores:", cv_scores)
        print("Mean R² Score from Cross-Validation:", np.mean(cv_scores))

    def plot_actual_vs_predicted(self, y_true, y_pred, dataset_type):
        """
        Plot actual vs predicted prices for model evaluation.

        Parameters:
        - y_true: Actual target values.
        - y_pred: Predicted target values.
        - dataset_type: Type of dataset (e.g., 'Training' or 'Testing').
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(
            y_true, y_pred, color="blue", alpha=0.7, label="Predicted vs Actual"
        )
        plt.plot(
            [y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            "g--",
            lw=2,
            label="Ideal Prediction",
        )
        plt.xlabel("Actual Prices")
        plt.ylabel("Predicted Prices")
        plt.title(f"Random Forest: Actual vs Predicted Prices ({dataset_type} Set)")
        plt.legend()
        plt.show()

    def feature_importance(self, X):
        """
        Analyze and plot feature importance using the trained Random Forest model.

        Parameters:
        - X: Feature data used for training.
        """
        feature_importances = self.rf_regressor.feature_importances_
        features_df = pd.DataFrame(
            {"Feature": X.columns, "Importance": feature_importances}
        )
        features_df = features_df.sort_values(by="Importance", ascending=False)

        print("\nFeature Importance:")
        print(features_df)

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(features_df["Feature"], features_df["Importance"], color="skyblue")
        plt.gca().invert_yaxis()
        plt.xlabel("Feature Importance")
        plt.title("Random Forest Feature Importance")
        plt.show()


def main():
    """
    Main function to load data, train the model, and evaluate its performance.
    """
    # Path to the data file
    data_path = "./data/immoweb_data_processed.csv"

    # Initialize Random Forest model
    rf_model = RandomForestModel(data_path)

    # Prepare data
    X, y = rf_model.prepare_data()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.22, random_state=42
    )

    # Train the model
    rf_model.train_model(X_train, y_train)

    # Make predictions
    pred_train = rf_model.predict(X_train)
    pred_test = rf_model.predict(X_test)

    # Evaluate the model
    evaluator = ModelEvaluator()
    evaluator.evaluate_model(y_train, pred_train, "Training")
    evaluator.evaluate_model(y_test, pred_test, "Testing")

    # Visualize predictions
    rf_model.plot_actual_vs_predicted(y_train, pred_train, "Training")
    rf_model.plot_actual_vs_predicted(y_test, pred_test, "Testing")

    # Cross-validation
    rf_model.cross_validate(X, y)

    # Feature importance analysis
    rf_model.feature_importance(X)


if __name__ == "__main__":
    main()
