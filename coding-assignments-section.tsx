"use client"

// Updated: [Current Date] - ML Assignments Section

import type React from "react"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import {
  Code,
  Upload,
  CheckCircle,
  Clock,
  FileText,
  Download,
  GitBranch,
  Lightbulb,
  AlertCircle,
  Trophy,
  Star,
  MessageSquare,
  ThumbsUp,
  ThumbsDown,
  Award,
  Send,
  Flag,
  Crown,
  Medal,
  Flame,
  FolderDown,
  FileUp,
  Laptop,
  Target,
  Brain,
  Ship,
  Home,
  Flower,
  HelpCircle,
  ExternalLink,
} from "lucide-react"
import { Action, Actions } from "@/components/documentation/actions"

const assignments = [
  {
    id: "titanic-classification",
    title: "Assignment 1: Titanic Survival Prediction (Binary Classification)",
    difficulty: "Beginner",
    estimatedTime: "3-4 hours",
    description:
      "Build a binary classification model to predict passenger survival on the Titanic using logistic regression and decision trees from scratch.",
    points: 200,
    dueDate: "2024-02-10",
    status: "not_started" as const,
    icon: Ship,
    category: "Binary Classification",

    learningObjectives: [
      "Implement logistic regression from scratch using gradient descent",
      "Build a decision tree classifier with entropy and information gain",
      "Handle missing values and categorical encoding",
      "Perform feature engineering on passenger data",
      "Evaluate binary classification models with precision, recall, F1-score",
      "Create data visualizations for exploratory analysis",
    ],

    datasets: [
      {
        name: "Titanic Training Dataset",
        filename: "titanic_train.csv",
        url: "https://www.kaggle.com/competitions/titanic/download/train.csv",
        kaggleUrl: "https://www.kaggle.com/competitions/titanic",
        description: "Training data with 891 passengers including survival labels",
        size: "60 KB",
        format: "CSV",
        samples: 891,
        features: [
          "PassengerId",
          "Survived",
          "Pclass",
          "Name",
          "Sex",
          "Age",
          "SibSp",
          "Parch",
          "Ticket",
          "Fare",
          "Cabin",
          "Embarked",
        ],
        target: "Survived",
      },
      {
        name: "Titanic Test Dataset",
        filename: "titanic_test.csv",
        url: "https://www.kaggle.com/competitions/titanic/download/test.csv",
        kaggleUrl: "https://www.kaggle.com/competitions/titanic",
        description: "Test data with 418 passengers for final predictions",
        size: "28 KB",
        format: "CSV",
        samples: 418,
      },
      {
        name: "Sample Submission",
        filename: "sample_submission.csv",
        url: "https://www.kaggle.com/competitions/titanic/download/gender_submission.csv",
        kaggleUrl: "https://www.kaggle.com/competitions/titanic",
        description: "Example submission format for Kaggle competition",
        size: "3 KB",
        format: "CSV",
      },
    ],

    codeExample: `# Titanic Survival Prediction - Binary Classification
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Example: Logistic Regression Implementation
class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        
    def sigmoid(self, z):
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.max_iterations):
            # Forward pass
            linear_pred = X @ self.weights + self.bias
            predictions = self.sigmoid(linear_pred)
            
            # Compute cost (log loss)
            cost = self.compute_cost(y, predictions)
            
            # Compute gradients
            dw = (1/n_samples) * X.T @ (predictions - y)
            db = (1/n_samples) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def compute_cost(self, y_true, y_pred):
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def predict(self, X):
        linear_pred = X @ self.weights + self.bias
        predictions = self.sigmoid(linear_pred)
        return (predictions >= 0.5).astype(int)
    
    def predict_proba(self, X):
        linear_pred = X @ self.weights + self.bias
        return self.sigmoid(linear_pred)

# Example usage:
# Load data
train_df = pd.read_csv('titanic_train.csv')

# Basic preprocessing
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})

# Feature selection
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
X = train_df[features].fillna(0)
y = train_df['Survived']

# Train model
model = LogisticRegression(learning_rate=0.01, max_iterations=1000)
model.fit(X.values, y.values)

# Make predictions
predictions = model.predict(X.values)
probabilities = model.predict_proba(X.values)

print(f"Training Accuracy: {np.mean(predictions == y.values):.4f}")`,

    mcqs: [
      {
        id: 1,
        question: "What is the activation function used in logistic regression?",
        options: ["ReLU function", "Sigmoid function", "Tanh function", "Linear function"],
        correct: 1,
        explanation:
          "Logistic regression uses the sigmoid function σ(z) = 1/(1 + e^(-z)) to map any real number to a value between 0 and 1, making it suitable for binary classification probabilities.",
      },
      {
        id: 2,
        question: "What is the cost function used in logistic regression?",
        options: ["Mean Squared Error", "Mean Absolute Error", "Log Loss (Cross-entropy)", "Hinge Loss"],
        correct: 2,
        explanation:
          "Logistic regression uses log loss (cross-entropy): -[y*log(p) + (1-y)*log(1-p)], which penalizes confident wrong predictions more heavily than uncertain predictions.",
      },
      {
        id: 3,
        question: "In the Titanic dataset, which feature is most likely to be the strongest predictor of survival?",
        options: ["PassengerId", "Ticket number", "Sex (Gender)", "Name length"],
        correct: 2,
        explanation:
          "Gender (Sex) is historically the strongest predictor in the Titanic dataset due to the 'women and children first' evacuation protocol, with women having much higher survival rates.",
      },
      {
        id: 4,
        question: "What does the 'Pclass' feature represent in the Titanic dataset?",
        options: [
          "Passenger age class",
          "Passenger ticket class (1st, 2nd, 3rd)",
          "Passenger survival class",
          "Passenger boarding class",
        ],
        correct: 1,
        explanation:
          "Pclass represents the passenger ticket class: 1 = First class (upper), 2 = Second class (middle), 3 = Third class (lower). Higher classes had better survival rates.",
      },
      {
        id: 5,
        question: "How should you handle missing values in the 'Age' column for the Titanic dataset?",
        options: [
          "Remove all rows with missing ages",
          "Fill with 0",
          "Fill with median age or use more sophisticated imputation",
          "Leave as NaN",
        ],
        correct: 2,
        explanation:
          "Filling with median age or using sophisticated imputation (like grouping by Pclass/Sex) preserves data and provides reasonable estimates. Removing rows loses valuable information.",
      },
    ],

    template: `# Titanic Survival Prediction - Binary Classification Assignment
# Student: [Your Name]
# Date: [Current Date]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

print("🚢 Titanic Survival Prediction - Binary Classification")
print("=" * 60)

# ============================================================================
# PART 1: DATA LOADING AND EXPLORATION
# ============================================================================

def load_titanic_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Titanic training and test datasets
    
    Returns:
        Tuple of (train_df, test_df)
    """
    # TODO: Load the datasets
    # train_df = pd.read_csv('titanic_train.csv')
    # test_df = pd.read_csv('titanic_test.csv')
    # return train_df, test_df
    pass

def explore_titanic_data(df: pd.DataFrame) -> None:
    """
    Perform comprehensive data exploration
    """
    print("\\n📊 Dataset Overview:")
    # TODO: Display basic information
    # print(f"Shape: {df.shape}")
    # print(f"\\nColumns: {df.columns.tolist()}")
    # print(f"\\nMissing values:\\n{df.isnull().sum()}")
    # print(f"\\nSurvival rate: {df['Survived'].mean():.3f}")
    
    # TODO: Create visualizations
    # - Survival rate by gender
    # - Survival rate by passenger class
    # - Age distribution by survival
    # - Correlation heatmap
    pass

# ============================================================================
# PART 2: DATA PREPROCESSING
# ============================================================================

class TitanicPreprocessor:
    """
    Handle Titanic-specific preprocessing
    """
    
    def __init__(self):
        self.age_median = None
        self.fare_median = None
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in Titanic dataset
        """
        # TODO: Handle missing values
        # Age: fill with median or group-based median
        # Fare: fill with median
        # Embarked: fill with mode
        # Cabin: create binary feature for cabin known/unknown
        pass
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features
        """
        # TODO: Encode categorical variables
        # Sex: male=0, female=1
        # Embarked: one-hot encoding or label encoding
        pass
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing ones
        """
        # TODO: Create new features
        # FamilySize = SibSp + Parch + 1
        # IsAlone = 1 if FamilySize == 1, else 0
        # Title extraction from Name (Mr, Mrs, Miss, etc.)
        # Age groups (Child, Adult, Senior)
        # Fare per person = Fare / FamilySize
        pass

# ============================================================================
# PART 3: MODEL IMPLEMENTATIONS
# ============================================================================

class LogisticRegression:
    """
    Logistic Regression implementation from scratch
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function
        """
        # TODO: Implement sigmoid function
        # Clip z to prevent overflow: z = np.clip(z, -500, 500)
        # return 1 / (1 + np.exp(-z))
        pass
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the logistic regression model
        """
        # TODO: Implement training using gradient descent
        # Initialize weights and bias
        # For each iteration:
        #   - Compute predictions using sigmoid
        #   - Compute cost (log loss)
        #   - Compute gradients
        #   - Update parameters
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make binary predictions
        """
        # TODO: Implement prediction
        # Use threshold of 0.5 for binary classification
        pass
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        """
        # TODO: Return probabilities using sigmoid
        pass

class DecisionTreeClassifier:
    """
    Decision Tree implementation from scratch
    """
    
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
    
    def entropy(self, y: np.ndarray) -> float:
        """
        Calculate entropy of a dataset
        """
        # TODO: Implement entropy calculation
        # Entropy = -Σ(p_i * log2(p_i))
        pass
    
    def information_gain(self, X: np.ndarray, y: np.ndarray, feature_idx: int, threshold: float) -> float:
        """
        Calculate information gain for a split
        """
        # TODO: Implement information gain
        # IG = entropy(parent) - weighted_avg(entropy(children))
        pass
    
    def find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
        """
        Find the best feature and threshold to split on
        """
        # TODO: Find best split by maximizing information gain
        pass
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Build the decision tree
        """
        # TODO: Implement recursive tree building
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the decision tree
        """
        # TODO: Implement prediction by traversing the tree
        pass

# ============================================================================
# PART 4: MODEL EVALUATION
# ============================================================================

class BinaryClassificationEvaluator:
    """
    Evaluation metrics for binary classification
    """
    
    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy"""
        # TODO: Implement accuracy = correct_predictions / total_predictions
        pass
    
    @staticmethod
    def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate precision"""
        # TODO: Implement precision = TP / (TP + FP)
        pass
    
    @staticmethod
    def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate recall (sensitivity)"""
        # TODO: Implement recall = TP / (TP + FN)
        pass
    
    @staticmethod
    def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate F1-score"""
        # TODO: Implement F1 = 2 * (precision * recall) / (precision + recall)
        pass
    
    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate confusion matrix"""
        # TODO: Implement 2x2 confusion matrix
        pass

# ============================================================================
# PART 5: MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution pipeline
    """
    print("\\n🚀 Starting Titanic Survival Prediction Pipeline...")
    
    # Step 1: Load Data
    print("\\n📁 Loading Titanic datasets...")
    # TODO: Load training and test data
    
    # Step 2: Explore Data
    print("\\n🔍 Exploring data...")
    # TODO: Perform data exploration
    
    # Step 3: Preprocess Data
    print("\\n🔧 Preprocessing data...")
    # TODO: Handle missing values, encode features, engineer new features
    
    # Step 4: Prepare Features and Target
    # TODO: Select features and prepare X, y
    
    # Step 5: Train Models
    print("\\n🤖 Training models...")
    
    # Logistic Regression
    print("\\n📊 Training Logistic Regression...")
    # TODO: Train logistic regression model
    
    # Decision Tree
    print("\\n🌳 Training Decision Tree...")
    # TODO: Train decision tree model
    
    # Step 6: Evaluate Models
    print("\\n📊 Evaluating models...")
    # TODO: Evaluate both models using all metrics
    
    # Step 7: Make Predictions on Test Set
    print("\\n🔮 Making predictions on test set...")
    # TODO: Preprocess test data and make predictions
    
    # Step 8: Create Submission File
    # TODO: Create submission file for Kaggle
    
    print("\\n✅ Pipeline completed successfully!")

if __name__ == "__main__":
    main()

# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

def test_implementations():
    """
    Test your implementations with simple data
    """
    # TODO: Create simple test cases
    pass

print("\\n📝 Titanic Classification Template loaded!")
print("Complete the TODO sections to implement your solution.")
print("Remember: No sklearn allowed - implement everything from scratch!")`,

    hints: [
      {
        id: 1,
        title: "Data Exploration Insights",
        content:
          "Focus on survival rates by gender, class, and age. Women and children had higher survival rates. First-class passengers survived more than third-class. Create visualizations to show these patterns.",
        unlocked: true,
      },
      {
        id: 2,
        title: "Handling Missing Ages",
        content:
          "Don't just use overall median for Age. Consider using median age by Pclass and Sex groups for better imputation. For example, median age of first-class women vs third-class men.",
        unlocked: false,
      },
      {
        id: 3,
        title: "Feature Engineering Ideas",
        content:
          "Extract titles from names (Mr, Mrs, Miss, Master). Create FamilySize = SibSp + Parch + 1. Create IsAlone binary feature. Consider age groups (Child < 16, Adult 16-64, Senior > 64).",
        unlocked: false,
      },
      {
        id: 4,
        title: "Logistic Regression Implementation",
        content:
          "Use gradient descent with sigmoid activation. Initialize weights to zeros. Use learning rate around 0.01. Clip z values in sigmoid to prevent overflow. Monitor cost function to ensure convergence.",
        unlocked: false,
      },
    ],
  },

  {
    id: "house-price-regression",
    title: "Assignment 2: House Price Prediction (Regression)",
    difficulty: "Intermediate",
    estimatedTime: "4-5 hours",
    description:
      "Build regression models to predict house prices using linear regression, polynomial regression, and regularization techniques from scratch.",
    points: 250,
    dueDate: "2024-02-17",
    status: "not_started" as const,
    icon: Home,
    category: "Regression",

    learningObjectives: [
      "Implement multiple regression algorithms from scratch",
      "Apply regularization techniques (Ridge, Lasso)",
      "Perform advanced feature engineering and selection",
      "Handle multicollinearity and feature scaling",
      "Implement cross-validation for model selection",
      "Create comprehensive regression evaluation metrics",
    ],

    datasets: [
      {
        name: "House Prices Training Data",
        filename: "house_prices_train.csv",
        url: "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/download/train.csv",
        kaggleUrl: "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques",
        description: "Training data with 1460 houses and 79 features",
        size: "460 KB",
        format: "CSV",
        samples: 1460,
        features: [
          "MSSubClass",
          "MSZoning",
          "LotFrontage",
          "LotArea",
          "Street",
          "Alley",
          "LotShape",
          "LandContour",
          "Utilities",
          "LotConfig",
        ],
        target: "SalePrice",
      },
      {
        name: "House Prices Test Data",
        filename: "house_prices_test.csv",
        url: "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/download/test.csv",
        kaggleUrl: "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques",
        description: "Test data with 1459 houses for predictions",
        size: "451 KB",
        format: "CSV",
        samples: 1459,
      },
      {
        name: "Data Description",
        filename: "data_description.txt",
        url: "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/download/data_description.txt",
        kaggleUrl: "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques",
        description: "Detailed description of all 79 features",
        size: "15 KB",
        format: "TXT",
      },
    ],

    codeExample: `# House Price Prediction - Regression Example
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Example: Ridge Regression Implementation
class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Regularization parameter
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        # Add bias term to X
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        # Ridge regression formula: (X^T X + αI)^(-1) X^T y
        XTX = X_with_bias.T @ X_with_bias
        
        # Add regularization term (don't regularize bias)
        regularization = self.alpha * np.eye(XTX.shape[0])
        regularization[0, 0] = 0  # Don't regularize bias term
        
        # Solve normal equation with regularization
        theta = np.linalg.inv(XTX + regularization) @ X_with_bias.T @ y
        
        self.bias = theta[0]
        self.weights = theta[1:]
    
    def predict(self, X):
        return X @ self.weights + self.bias

# Example: Feature Engineering for House Prices
def engineer_features(df):
    # Create new features
    df['TotalSF'] = df['1stFlrSF'] + df['2ndFlrSF'] + df['TotalBsmtSF']
    df['TotalBathrooms'] = df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
    df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
    
    # Age of house
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    
    # Quality scores
    df['OverallScore'] = df['OverallQual'] * df['OverallCond']
    
    return df

# Example usage:
# Load and preprocess data
train_df = pd.read_csv('house_prices_train.csv')
train_df = engineer_features(train_df)

# Select numerical features for simple example
numerical_features = ['GrLivArea', 'TotalBsmtSF', 'GarageArea', 'TotalSF']
X = train_df[numerical_features].fillna(0)
y = train_df['SalePrice']

# Train Ridge regression
model = RidgeRegression(alpha=1.0)
model.fit(X.values, y.values)

# Make predictions
predictions = model.predict(X.values)
mse = np.mean((y.values - predictions) ** 2)
print(f"Mean Squared Error: {mse:.2f}")`,

    mcqs: [
      {
        id: 1,
        question: "What is the main difference between Ridge and Lasso regression?",
        options: [
          "Ridge uses L1 penalty, Lasso uses L2 penalty",
          "Ridge uses L2 penalty, Lasso uses L1 penalty",
          "Ridge is for classification, Lasso is for regression",
          "There is no difference",
        ],
        correct: 1,
        explanation:
          "Ridge regression uses L2 penalty (sum of squared coefficients) which shrinks coefficients toward zero. Lasso uses L1 penalty (sum of absolute coefficients) which can set coefficients exactly to zero, performing feature selection.",
      },
      {
        id: 2,
        question: "What does R-squared (coefficient of determination) measure?",
        options: [
          "The correlation between features",
          "The proportion of variance in the target explained by the model",
          "The mean squared error of predictions",
          "The number of features in the model",
        ],
        correct: 1,
        explanation:
          "R-squared measures the proportion of variance in the dependent variable that is predictable from the independent variables. R² = 1 - (SS_res / SS_tot), where values closer to 1 indicate better model fit.",
      },
      {
        id: 3,
        question: "Why is feature scaling important for regularized regression models?",
        options: [
          "It makes the model train faster",
          "It prevents overfitting",
          "Regularization penalties are scale-dependent",
          "It improves R-squared",
        ],
        correct: 2,
        explanation:
          "Regularization penalties (L1/L2) are scale-dependent. Features with larger scales will be penalized more heavily than features with smaller scales, leading to unfair regularization. Scaling ensures equal treatment.",
      },
      {
        id: 4,
        question: "What is multicollinearity and why is it problematic?",
        options: [
          "When features are categorical",
          "When features are highly correlated with each other",
          "When the target variable is continuous",
          "When there are missing values",
        ],
        correct: 1,
        explanation:
          "Multicollinearity occurs when independent variables are highly correlated. It makes coefficient estimates unstable and difficult to interpret, and can cause numerical instability in matrix inversion.",
      },
      {
        id: 5,
        question: "In polynomial regression, what happens as you increase the degree?",
        options: [
          "The model becomes simpler",
          "Training error increases, test error decreases",
          "Training error decreases, but test error may increase (overfitting)",
          "Both training and test errors always decrease",
        ],
        correct: 2,
        explanation:
          "Higher degree polynomials can fit training data more closely (lower training error) but may overfit, leading to poor generalization and higher test error. This is the bias-variance tradeoff.",
      },
    ],

    template: `# House Price Prediction - Regression Assignment
# Student: [Your Name]  
# Date: [Current Date]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

print("🏠 House Price Prediction - Regression Analysis")
print("=" * 60)

# ============================================================================
# PART 1: DATA LOADING AND EXPLORATION
# ============================================================================

def load_house_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load house price training and test datasets
    """
    # TODO: Load the datasets
    # train_df = pd.read_csv('house_prices_train.csv')
    # test_df = pd.read_csv('house_prices_test.csv')
    # return train_df, test_df
    pass

def explore_house_data(df: pd.DataFrame) -> None:
    """
    Comprehensive exploration of house price data
    """
    print("\\n📊 Dataset Overview:")
    # TODO: Basic data exploration
    # print(f"Shape: {df.shape}")
    # print(f"\\nTarget variable statistics:")
    # print(df['SalePrice'].describe())
    
    # TODO: Create visualizations
    # - SalePrice distribution (histogram, log-transform)
    # - Correlation heatmap of top features
    # - Scatter plots of key features vs SalePrice
    # - Missing values heatmap
    pass

# ============================================================================
# PART 2: DATA PREPROCESSING AND FEATURE ENGINEERING
# ============================================================================

class HousePricePreprocessor:
    """
    Specialized preprocessing for house price data
    """
    
    def __init__(self):
        self.feature_means = {}
        self.feature_stds = {}
        self.categorical_encoders = {}
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in house price dataset
        """
        # TODO: Handle missing values strategically
        # Numerical: median/mean imputation or domain-specific values
        # Categorical: mode imputation or 'None' category
        # Some missing values are meaningful (e.g., no garage = 0 garage area)
        pass
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features
        """
        # TODO: Handle categorical encoding
        # Ordinal features: map to numerical values (e.g., ExterQual: Ex=5, Gd=4, TA=3, Fa=2, Po=1)
        # Nominal features: one-hot encoding or target encoding
        pass
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing ones
        """
        # TODO: Engineer meaningful features
        # Total square footage combinations
        # Age-related features (house age, remodel age)
        # Quality scores and interactions
        # Bathroom and room counts
        # Porch and outdoor space totals
        pass
    
    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and handle outliers
        """
        # TODO: Handle outliers in both features and target
        # Use IQR method or domain knowledge
        # Consider log transformation for skewed variables
        pass
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features
        """
        # TODO: Implement feature scaling
        # Use standardization for regularized models
        pass

# ============================================================================
# PART 3: REGRESSION MODEL IMPLEMENTATIONS
# ============================================================================

class LinearRegression:
    """
    Linear Regression using Normal Equation
    """
    
    def __init__(self):
        self.weights = None
        self.bias = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit linear regression using normal equation
        """
        # TODO: Implement normal equation
        # θ = (X^T * X)^(-1) * X^T * y
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        """
        # TODO: Implement prediction
        pass

class RidgeRegression:
    """
    Ridge Regression with L2 regularization
    """
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.weights = None
        self.bias = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit Ridge regression
        """
        # TODO: Implement Ridge regression
        # θ = (X^T * X + αI)^(-1) * X^T * y
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        """
        # TODO: Implement prediction
        pass

class LassoRegression:
    """
    Lasso Regression with L1 regularization
    """
    
    def __init__(self, alpha: float = 1.0, max_iterations: int = 1000):
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
    
    def soft_threshold(self, x: float, threshold: float) -> float:
        """
        Soft thresholding function for Lasso
        """
        # TODO: Implement soft thresholding
        # if x > threshold: return x - threshold
        # elif x < -threshold: return x + threshold
        # else: return 0
        pass
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit Lasso regression using coordinate descent
        """
        # TODO: Implement Lasso using coordinate descent
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        """
        # TODO: Implement prediction
        pass

class PolynomialRegression:
    """
    Polynomial Regression
    """
    
    def __init__(self, degree: int = 2):
        self.degree = degree
        self.linear_model = LinearRegression()
    
    def create_polynomial_features(self, X: np.ndarray) -> np.ndarray:
        """
        Create polynomial features
        """
        # TODO: Create polynomial features up to specified degree
        # Include interaction terms for degree > 1
        pass
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit polynomial regression
        """
        # TODO: Create polynomial features and fit linear model
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        """
        # TODO: Transform features and predict
        pass

# ============================================================================
# PART 4: MODEL EVALUATION
# ============================================================================

class RegressionEvaluator:
    """
    Comprehensive regression evaluation metrics
    """
    
    @staticmethod
    def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate MSE"""
        # TODO: Implement MSE
        pass
    
    @staticmethod
    def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate RMSE"""
        # TODO: Implement RMSE
        pass
    
    @staticmethod
    def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate MAE"""
        # TODO: Implement MAE
        pass
    
    @staticmethod
    def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared"""
        # TODO: Implement R²
        pass
    
    @staticmethod
    def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate MAPE"""
        # TODO: Implement MAPE
        # MAPE = mean(|y_true - y_pred| / |y_true|) * 100
        pass
    
    def cross_validate(self, model, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, List[float]]:
        """
        Perform k-fold cross-validation
        """
        # TODO: Implement k-fold cross-validation
        pass

# ============================================================================
# PART 5: MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution pipeline
    """
    print("\\n🚀 Starting House Price Prediction Pipeline...")
    
    # Step 1: Load Data
    print("\\n📁 Loading house price datasets...")
    # TODO: Load data
    
    # Step 2: Explore Data  
    print("\\n🔍 Exploring data...")
    # TODO: Perform comprehensive EDA
    
    # Step 3: Preprocess Data
    print("\\n🔧 Preprocessing data...")
    # TODO: Handle missing values, encode features, engineer features
    
    # Step 4: Train Multiple Models
    print("\\n🤖 Training regression models...")
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': RidgeRegression(alpha=1.0),
        'Lasso Regression': LassoRegression(alpha=1.0),
        'Polynomial Regression': PolynomialRegression(degree=2)
    }
    
    # TODO: Train and evaluate each model
    
    # Step 5: Model Selection and Hyperparameter Tuning
    print("\\n🎯 Model selection and tuning...")
    # TODO: Use cross-validation to select best model and hyperparameters
    
    # Step 6: Final Predictions
    print("\\n🔮 Making final predictions...")
    # TODO: Train best model on full training set and predict on test set
    
    print("\\n✅ Pipeline completed successfully!")

if __name__ == "__main__":
    main()

print("\\n📝 House Price Regression Template loaded!")
print("Implement all TODO sections for complete solution.")`,

    hints: [
      {
        id: 1,
        title: "Feature Engineering Strategy",
        content:
          "Focus on creating meaningful combinations: TotalSF (all square footages), TotalBathrooms, HouseAge, OverallScore (Qual*Cond). Log-transform skewed features like SalePrice and GrLivArea.",
        unlocked: true,
      },
      {
        id: 2,
        title: "Handling Missing Values",
        content:
          "Many missing values are meaningful (e.g., no garage = 0 garage area). For LotFrontage, use median by Neighborhood. For categorical features like Alley, 'None' might be the right value.",
        unlocked: false,
      },
      {
        id: 3,
        title: "Regularization Parameter Selection",
        content:
          "Use cross-validation to select alpha for Ridge/Lasso. Try values like [0.1, 1, 10, 100]. Ridge keeps all features, Lasso performs feature selection by setting coefficients to zero.",
        unlocked: false,
      },
      {
        id: 4,
        title: "Polynomial Features",
        content:
          "Be careful with polynomial degree - higher degrees can cause overfitting. Consider only degree 2 with selected features. Include interaction terms between important features like GrLivArea and OverallQual.",
        unlocked: false,
      },
    ],
  },

  {
    id: "iris-multiclass",
    title: "Assignment 3: Iris Species Classification (Multi-class)",
    difficulty: "Intermediate",
    estimatedTime: "3-4 hours",
    description:
      "Build multi-class classification models to classify iris flowers using k-nearest neighbors, naive Bayes, and neural networks from scratch.",
    points: 200,
    dueDate: "2024-02-24",
    status: "not_started" as const,
    icon: Flower,
    category: "Multi-class Classification",

    learningObjectives: [
      "Implement k-nearest neighbors (KNN) from scratch",
      "Build Gaussian Naive Bayes classifier",
      "Create a simple neural network with backpropagation",
      "Handle multi-class classification evaluation",
      "Perform dimensionality reduction and visualization",
      "Compare different distance metrics and algorithms",
    ],

    datasets: [
      {
        name: "Iris Dataset",
        filename: "iris.csv",
        url: "https://www.kaggle.com/datasets/uciml/iris/download",
        kaggleUrl: "https://www.kaggle.com/datasets/uciml/iris",
        description: "Classic iris dataset with 150 samples and 4 features",
        size: "4 KB",
        format: "CSV",
        samples: 150,
        features: ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        target: "species",
      },
      {
        name: "Extended Iris Dataset",
        filename: "iris_extended.csv",
        url: "https://www.kaggle.com/datasets/arshid/iris-flower-dataset/download",
        kaggleUrl: "https://www.kaggle.com/datasets/arshid/iris-flower-dataset",
        description: "Extended iris dataset with additional features and samples",
        size: "8 KB",
        format: "CSV",
        samples: 300,
      },
    ],

    codeExample: `# Iris Multi-class Classification Example
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Example: K-Nearest Neighbors Implementation
class KNearestNeighbors:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
    
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = []
        for x in X:
            # Calculate distances to all training points
            distances = []
            for i, x_train in enumerate(self.X_train):
                if self.distance_metric == 'euclidean':
                    dist = self.euclidean_distance(x, x_train)
                else:
                    dist = self.manhattan_distance(x, x_train)
                distances.append((dist, self.y_train[i]))
            
            # Sort by distance and get k nearest neighbors
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:self.k]
            
            # Vote for class (majority vote)
            votes = {}
            for _, label in k_nearest:
                votes[label] = votes.get(label, 0) + 1
            
            # Predict class with most votes
            predicted_class = max(votes, key=votes.get)
            predictions.append(predicted_class)
        
        return np.array(predictions)

# Example: Simple Neural Network
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Initialize weights randomly
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        m = X.shape[0]
        
        # Convert y to one-hot encoding
        y_one_hot = np.eye(len(np.unique(y)))[y]
        
        # Backward propagation
        dz2 = output - y_one_hot
        dW2 = (1/m) * self.a1.T @ dz2
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = (1/m) * X.T @ dz1
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

# Example usage:
# Load iris data
iris_df = pd.read_csv('iris.csv')
X = iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y = iris_df['species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2}).values

# Train KNN
knn = KNearestNeighbors(k=3)
knn.fit(X, y)
predictions = knn.predict(X)
accuracy = np.mean(predictions == y)
print(f"KNN Accuracy: {accuracy:.4f}")`,

    mcqs: [
      {
        id: 1,
        question: "In k-nearest neighbors (KNN), what happens when k=1?",
        options: [
          "The model becomes more robust to noise",
          "The decision boundary becomes smoother",
          "The model memorizes the training data (overfitting)",
          "The model becomes more generalizable",
        ],
        correct: 2,
        explanation:
          "When k=1, KNN simply assigns the class of the nearest neighbor, which can lead to overfitting as the model memorizes training data and creates very complex decision boundaries sensitive to noise.",
      },
      {
        id: 2,
        question: "What is the key assumption of Naive Bayes classifier?",
        options: [
          "Features are normally distributed",
          "Features are independent given the class",
          "Classes are equally likely",
          "Data is linearly separable",
        ],
        correct: 1,
        explanation:
          "Naive Bayes assumes conditional independence of features given the class label. This 'naive' assumption allows us to compute P(features|class) as the product of individual P(feature_i|class).",
      },
      {
        id: 3,
        question: "In multi-class classification, what does the softmax function do?",
        options: [
          "Converts outputs to binary values",
          "Normalizes outputs to sum to 1 (probability distribution)",
          "Applies threshold to outputs",
          "Computes the maximum output",
        ],
        correct: 1,
        explanation:
          "Softmax converts raw output scores into a probability distribution where all values are between 0 and 1 and sum to 1, making it suitable for multi-class probability estimation.",
      },
      {
        id: 4,
        question: "What is the curse of dimensionality in KNN?",
        options: [
          "KNN becomes faster with more dimensions",
          "Distance becomes less meaningful in high dimensions",
          "KNN requires more memory",
          "KNN becomes more accurate",
        ],
        correct: 1,
        explanation:
          "In high dimensions, all points tend to be roughly equidistant from each other, making distance-based similarity less meaningful and reducing KNN's effectiveness.",
      },
      {
        id: 5,
        question: "In the iris dataset, which feature is most useful for separating setosa from other species?",
        options: ["Sepal length", "Sepal width", "Petal length", "Petal width"],
        correct: 2,
        explanation:
          "Petal length (and petal width) are the most discriminative features for separating setosa from versicolor and virginica. Setosa has much smaller petals compared to the other two species.",
      },
    ],

    template: `# Iris Species Classification - Multi-class Classification Assignment
# Student: [Your Name]
# Date: [Current Date]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

print("🌸 Iris Species Classification - Multi-class Classification")
print("=" * 60)

# ============================================================================
# PART 1: DATA LOADING AND EXPLORATION
# ============================================================================

def load_iris_data() -> pd.DataFrame:
    """
    Load iris dataset
    """
    # TODO: Load the iris dataset
    # df = pd.read_csv('iris.csv')
    # return df
    pass

def explore_iris_data(df: pd.DataFrame) -> None:
    """
    Comprehensive exploration of iris data
    """
    print("\\n📊 Dataset Overview:")
    # TODO: Basic data exploration
    # print(f"Shape: {df.shape}")
    # print(f"\\nClasses: {df['species'].unique()}")
    # print(f"\\nClass distribution:\\n{df['species'].value_counts()}")
    
    # TODO: Create visualizations
    # - Pairplot colored by species
    # - Box plots for each feature by species
    # - Correlation heatmap
    # - 2D scatter plots of most discriminative features
    pass

# ============================================================================
# PART 2: DATA PREPROCESSING
# ============================================================================

class IrisPreprocessor:
    """
    Preprocessing for iris dataset
    """
    
    def __init__(self):
        self.feature_means = {}
        self.feature_stds = {}
        self.label_encoder = {}
    
    def encode_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode species labels to numerical values
        """
        # TODO: Encode species to numbers
        # setosa: 0, versicolor: 1, virginica: 2
        pass
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale features using standardization
        """
        # TODO: Standardize features for algorithms that are sensitive to scale
        pass
    
    def train_test_split(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.3, random_state: int = 42) -> Tuple:
        """
        Split data into training and testing sets
        """
        # TODO: Implement stratified train-test split
        # Ensure each class is represented proportionally in both sets
        pass

# ============================================================================
# PART 3: MODEL IMPLEMENTATIONS
# ============================================================================

class KNearestNeighbors:
    """
    K-Nearest Neighbors implementation from scratch
    """
    
    def __init__(self, k: int = 3, distance_metric: str = 'euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
    
    def euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two points
        """
        # TODO: Implement Euclidean distance
        # distance = sqrt(sum((x1_i - x2_i)^2))
        pass
    
    def manhattan_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Calculate Manhattan distance between two points
        """
        # TODO: Implement Manhattan distance
        # distance = sum(|x1_i - x2_i|)
        pass
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Store training data (lazy learning)
        """
        # TODO: Store training data
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict classes for test data
        """
        # TODO: For each test point:
        # 1. Calculate distances to all training points
        # 2. Find k nearest neighbors
        # 3. Use majority vote to predict class
        pass

class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes implementation from scratch
    """
    
    def __init__(self):
        self.class_priors = {}
        self.feature_means = {}
        self.feature_vars = {}
        self.classes = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train Naive Bayes classifier
        """
        # TODO: Calculate class priors and feature statistics
        # For each class:
        #   - Calculate prior probability P(class)
        #   - Calculate mean and variance for each feature P(feature|class)
        pass
    
    def gaussian_probability(self, x: float, mean: float, var: float) -> float:
        """
        Calculate Gaussian probability density
        """
        # TODO: Implement Gaussian PDF
        # P(x) = (1/sqrt(2*pi*var)) * exp(-0.5 * ((x-mean)^2 / var))
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict classes using Bayes theorem
        """
        # TODO: For each test point:
        # 1. Calculate posterior probability for each class
        # 2. P(class|features) ∝ P(class) * ∏P(feature_i|class)
        # 3. Predict class with highest posterior
        pass

class SimpleNeuralNetwork:
    """
    Simple Neural Network for multi-class classification
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, learning_rate: float = 0.01):
        # TODO: Initialize weights and biases randomly
        # Use small random values for weights, zeros for biases
        self.learning_rate = learning_rate
        pass
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """
        ReLU activation function
        """
        # TODO: Implement ReLU: max(0, x)
        pass
    
    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Derivative of ReLU
        """
        # TODO: Implement ReLU derivative: 1 if x > 0, else 0
        pass
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Softmax activation for output layer
        """
        # TODO: Implement softmax
        # softmax(x_i) = exp(x_i) / sum(exp(x_j))
        # Subtract max for numerical stability
        pass
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation
        """
        # TODO: Implement forward pass
        # Layer 1: z1 = X @ W1 + b1, a1 = relu(z1)
        # Layer 2: z2 = a1 @ W2 + b2, a2 = softmax(z2)
        pass
    
    def backward(self, X: np.ndarray, y: np.ndarray, output: np.ndarray) -> None:
        """
        Backward propagation
        """
        # TODO: Implement backpropagation
        # Calculate gradients and update weights
        pass
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000) -> None:
        """
        Train the neural network
        """
        # TODO: Training loop
        # For each epoch: forward pass, backward pass, update weights
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        """
        # TODO: Forward pass and return predicted classes
        pass

# ============================================================================
# PART 4: MODEL EVALUATION
# ============================================================================

class MultiClassEvaluator:
    """
    Evaluation metrics for multi-class classification
    """
    
    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy"""
        # TODO: Implement accuracy
        pass
    
    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
        """Calculate confusion matrix"""
        # TODO: Implement confusion matrix
        # matrix[i][j] = number of samples with true class i predicted as class j
        pass
    
    @staticmethod
    def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict:
        """Calculate precision, recall, F1 for each class"""
        # TODO: Calculate metrics for each class
        # Precision = TP / (TP + FP)
        # Recall = TP / (TP + FN)  
        # F1 = 2 * (precision * recall) / (precision + recall)
        pass
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, class_names: List[str]) -> None:
        """Plot confusion matrix"""
        # TODO: Create heatmap visualization of confusion matrix
        pass

# ============================================================================
# PART 5: DIMENSIONALITY REDUCTION AND VISUALIZATION
# ============================================================================

class PCA:
    """
    Principal Component Analysis implementation
    """
    
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.components = None
        self.mean = None
    
    def fit(self, X: np.ndarray) -> None:
        """
        Fit PCA to data
        """
        # TODO: Implement PCA
        # 1. Center the data (subtract mean)
        # 2. Calculate covariance matrix
        # 3. Find eigenvalues and eigenvectors
        # 4. Sort by eigenvalues and select top n_components
        pass
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to lower dimensions
        """
        # TODO: Project data onto principal components
        pass

# ============================================================================
# PART 6: MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution pipeline
    """
    print("\\n🚀 Starting Iris Classification Pipeline...")
    
    # Step 1: Load Data
    print("\\n📁 Loading iris dataset...")
    # TODO: Load data
    
    # Step 2: Explore Data
    print("\\n🔍 Exploring data...")
    # TODO: Perform EDA
    
    # Step 3: Preprocess Data
    print("\\n🔧 Preprocessing data...")
    # TODO: Encode labels, scale features, split data
    
    # Step 4: Train Models
    print("\\n🤖 Training classification models...")
    
    models = {
        'KNN (k=3)': KNearestNeighbors(k=3),
        'KNN (k=5)': KNearestNeighbors(k=5),
        'Naive Bayes': GaussianNaiveBayes(),
        'Neural Network': SimpleNeuralNetwork(input_size=4, hidden_size=8, output_size=3)
    }
    
    # TODO: Train and evaluate each model
    
    # Step 5: Model Comparison
    print("\\n📊 Comparing model performance...")
    # TODO: Compare accuracies, confusion matrices, etc.
    
    # Step 6: Dimensionality Reduction and Visualization
    print("\\n📈 Visualizing results...")
    # TODO: Apply PCA and create 2D visualizations
    
    # Step 7: Feature Importance Analysis
    print("\\n🔍 Analyzing feature importance...")
    # TODO: Analyze which features are most important for classification
    
    print("\\n✅ Pipeline completed successfully!")

if __name__ == "__main__":
    main()

# ============================================================================
# ADDITIONAL ANALYSIS FUNCTIONS
# ============================================================================

def compare_distance_metrics():
    """
    Compare different distance metrics for KNN
    """
    # TODO: Compare Euclidean vs Manhattan distance
    pass

def analyze_k_values():
    """
    Analyze effect of different k values in KNN
    """
    # TODO: Plot accuracy vs k value
    pass

def visualize_decision_boundaries():
    """
    Visualize decision boundaries for different classifiers
    """
    # TODO: Create 2D plots showing decision boundaries
    pass

print("\\n📝 Iris Multi-class Classification Template loaded!")
print("Complete all TODO sections to implement your solution.")
print("Focus on understanding the differences between algorithms!")`,

    hints: [
      {
        id: 1,
        title: "Understanding the Iris Dataset",
        content:
          "The iris dataset has 3 classes with 50 samples each. Setosa is linearly separable from the other two, but versicolor and virginica overlap. Petal measurements are more discriminative than sepal measurements.",
        unlocked: true,
      },
      {
        id: 2,
        title: "KNN Implementation Tips",
        content:
          "For KNN, try different k values (odd numbers to avoid ties). Use cross-validation to find optimal k. Consider the curse of dimensionality - with only 4 features, this isn't a problem for iris.",
        unlocked: false,
      },
      {
        id: 3,
        title: "Naive Bayes Assumptions",
        content:
          "Gaussian Naive Bayes assumes features are normally distributed and independent. Check these assumptions with histograms and correlation analysis. The independence assumption is often violated but the algorithm still works well.",
        unlocked: false,
      },
      {
        id: 4,
        title: "Neural Network Architecture",
        content:
          "For iris classification, a simple network with 4 inputs, 8 hidden units, and 3 outputs works well. Use ReLU activation in hidden layer and softmax in output. Start with learning rate 0.01.",
        unlocked: false,
      },
    ],
  },
]

// Mock peer review data
const peerReviews = [
  {
    id: 1,
    reviewer: "Sarah Chen",
    avatar: "/placeholder.svg?height=32&width=32&text=SC",
    rating: 5,
    comment:
      "Excellent Titanic implementation! Your feature engineering with family size and title extraction is very creative. The logistic regression convergence is smooth. Great visualizations showing survival patterns.",
    timestamp: "2 hours ago",
    helpful: 18,
    assignmentId: "titanic-classification",
  },
  {
    id: 2,
    reviewer: "Mike Rodriguez",
    avatar: "/placeholder.svg?height=32&width=32&text=MR",
    rating: 4,
    comment:
      "Solid house price prediction! Your Ridge regression implementation handles multicollinearity well. The polynomial features are well-chosen. Consider adding more sophisticated outlier detection.",
    timestamp: "1 day ago",
    helpful: 14,
    assignmentId: "house-price-regression",
  },
  {
    id: 3,
    reviewer: "Emma Wilson",
    avatar: "/placeholder.svg?height=32&width=32&text=EW",
    rating: 5,
    comment:
      "Beautiful iris classification! Your KNN implementation with different distance metrics is thorough. The neural network backpropagation is mathematically correct. Love the PCA visualization.",
    timestamp: "2 days ago",
    helpful: 16,
    assignmentId: "iris-multiclass",
  },
]

// Gamification data
const achievements = [
  {
    id: 1,
    name: "Classification Master",
    description: "Complete binary classification assignment",
    icon: "🎯",
    unlocked: false,
  },
  { id: 2, name: "Regression Expert", description: "Complete regression assignment", icon: "📈", unlocked: false },
  {
    id: 3,
    name: "Multi-class Champion",
    description: "Complete multi-class classification",
    icon: "🌸",
    unlocked: false,
  },
  {
    id: 4,
    name: "Algorithm Architect",
    description: "Implement 5+ algorithms from scratch",
    icon: "🏗️",
    unlocked: false,
  },
  { id: 5, name: "Data Visualizer", description: "Create 10+ meaningful visualizations", icon: "📊", unlocked: false },
  { id: 6, name: "Perfect Scorer", description: "Get 95%+ on any assignment", icon: "💯", unlocked: false },
]

const leaderboard = [
  { rank: 1, name: "Alex Thompson", points: 580, streak: 15, avatar: "/placeholder.svg?height=32&width=32&text=AT" },
  { rank: 2, name: "Maria Garcia", points: 520, streak: 12, avatar: "/placeholder.svg?height=32&width=32&text=MG" },
  { rank: 3, name: "David Kim", points: 480, streak: 8, avatar: "/placeholder.svg?height=32&width=32&text=DK" },
  { rank: 4, name: "You", points: 0, streak: 0, avatar: "/placeholder.svg?height=32&width=32&text=YU" },
  { rank: 5, name: "Lisa Wang", points: 420, streak: 6, avatar: "/placeholder.svg?height=32&width=32&text=LW" },
]

export function CodingAssignmentsSection() {
  const [activeAssignment, setActiveAssignment] = useState(assignments[0].id)
  const [currentVideoId, setCurrentVideoId] = useState<string | null>(null)
  const [isVideoPlaying, setIsVideoPlaying] = useState(false)
  const [videoMuted, setVideoMuted] = useState(false)
  const [newReview, setNewReview] = useState("")
  const [reviewRating, setReviewRating] = useState(5)
  const [uploadedFiles, setUploadedFiles] = useState<Record<string, File[]>>({})
  const [assignmentStates, setAssignmentStates] = useState<
    Record<
      string,
      {
        status: "not_started" | "in_progress" | "submitted" | "completed"
        unlockedHints: number
        submissionFeedback: string
        completionPercentage: number
        submittedFiles: string[]
        submissionDate: string | null
        mcqAnswers: Record<number, number>
        mcqScore: number
      }
    >
  >(() => {
    const initialStates: Record<string, any> = {}
    assignments.forEach((assignment) => {
      initialStates[assignment.id] = {
        status: "not_started",
        unlockedHints: 1,
        submissionFeedback: "",
        completionPercentage: 0,
        submittedFiles: [],
        submissionDate: null,
        mcqAnswers: {},
        mcqScore: 0,
      }
    })
    return initialStates
  })

  const [autoGradeResults, setAutoGradeResults] = useState<
    Record<
      string,
      {
        score: number
        maxScore: number
        testResults: Array<{
          testName: string
          passed: boolean
          points: number
          maxPoints: number
          feedback: string
          executionTime?: number
        }>
        codeQuality: {
          score: number
          issues: string[]
          suggestions: string[]
        }
        overallFeedback: string
        gradedAt: string
      }
    >
  >({})

  const currentAssignment = assignments.find((a) => a.id === activeAssignment)!
  const currentState = assignmentStates[activeAssignment]

  const unlockNextHint = () => {
    const maxHints = currentAssignment.hints.length
    if (currentState.unlockedHints < maxHints) {
      setAssignmentStates((prev) => ({
        ...prev,
        [activeAssignment]: {
          ...prev[activeAssignment],
          unlockedHints: prev[activeAssignment].unlockedHints + 1,
        },
      }))
    }
  }

  const handleMCQAnswer = (questionId: number, answerIndex: number) => {
    setAssignmentStates((prev) => ({
      ...prev,
      [activeAssignment]: {
        ...prev[activeAssignment],
        mcqAnswers: {
          ...prev[activeAssignment].mcqAnswers,
          [questionId]: answerIndex,
        },
      },
    }))
  }

  const calculateMCQScore = () => {
    const answers = currentState.mcqAnswers
    let correct = 0
    currentAssignment.mcqs.forEach((mcq) => {
      if (answers[mcq.id] === mcq.correct) {
        correct++
      }
    })
    const score = (correct / currentAssignment.mcqs.length) * 100

    setAssignmentStates((prev) => ({
      ...prev,
      [activeAssignment]: {
        ...prev[activeAssignment],
        mcqScore: score,
      },
    }))

    return score
  }

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || [])
    if (files.length > 0) {
      setUploadedFiles((prev) => ({
        ...prev,
        [activeAssignment]: [...(prev[activeAssignment] || []), ...files],
      }))

      setAssignmentStates((prev) => ({
        ...prev,
        [activeAssignment]: {
          ...prev[activeAssignment],
          status: "in_progress",
          completionPercentage: Math.min(100, prev[activeAssignment].completionPercentage + 25),
        },
      }))
    }
  }

  const runAutoGrader = async (files: File[], assignmentId: string) => {
    // Simulate auto-grading process
    await new Promise((resolve) => setTimeout(resolve, 3000))

    // Define test cases for each assignment
    const testCases = {
      "titanic-classification": [
        {
          testName: "Logistic Regression Implementation",
          passed: true,
          points: 35,
          maxPoints: 40,
          feedback: "Good sigmoid and gradient descent implementation. Consider adding convergence checking.",
          executionTime: 120,
        },
        {
          testName: "Decision Tree Implementation",
          passed: true,
          points: 30,
          maxPoints: 35,
          feedback: "Entropy and information gain correctly implemented. Tree building logic is sound.",
          executionTime: 85,
        },
        {
          testName: "Data Preprocessing",
          passed: true,
          points: 25,
          maxPoints: 30,
          feedback: "Good handling of missing values and feature engineering. Title extraction is creative.",
          executionTime: 45,
        },
        {
          testName: "Model Evaluation",
          passed: true,
          points: 20,
          maxPoints: 25,
          feedback: "All classification metrics correctly implemented. Confusion matrix visualization is clear.",
          executionTime: 30,
        },
      ],
      "house-price-regression": [
        {
          testName: "Linear Regression Implementation",
          passed: true,
          points: 25,
          maxPoints: 30,
          feedback: "Normal equation correctly implemented. Good handling of matrix inversion edge cases.",
          executionTime: 60,
        },
        {
          testName: "Ridge Regression Implementation",
          passed: true,
          points: 28,
          maxPoints: 30,
          feedback: "Excellent regularization implementation. Alpha parameter handling is correct.",
          executionTime: 70,
        },
        {
          testName: "Lasso Regression Implementation",
          passed: false,
          points: 15,
          maxPoints: 30,
          feedback: "Coordinate descent algorithm needs improvement. Soft thresholding is partially correct.",
          executionTime: 90,
        },
        {
          testName: "Feature Engineering",
          passed: true,
          points: 25,
          maxPoints: 30,
          feedback: "Creative feature combinations. TotalSF and age-related features are well-designed.",
          executionTime: 40,
        },
        {
          testName: "Cross-Validation",
          passed: true,
          points: 22,
          maxPoints: 25,
          feedback: "K-fold implementation is correct. Good use of stratification for regression.",
          executionTime: 110,
        },
      ],
      "iris-multiclass": [
        {
          testName: "KNN Implementation",
          passed: true,
          points: 30,
          maxPoints: 35,
          feedback: "Both Euclidean and Manhattan distances correctly implemented. Voting mechanism works well.",
          executionTime: 50,
        },
        {
          testName: "Naive Bayes Implementation",
          passed: true,
          points: 32,
          maxPoints: 35,
          feedback: "Gaussian probability calculations are accurate. Prior and likelihood estimation is correct.",
          executionTime: 40,
        },
        {
          testName: "Neural Network Implementation",
          passed: true,
          points: 25,
          maxPoints: 30,
          feedback: "Forward and backward propagation correctly implemented. Softmax and ReLU functions work properly.",
          executionTime: 150,
        },
        {
          testName: "PCA Implementation",
          passed: true,
          points: 20,
          maxPoints: 25,
          feedback: "Eigenvalue decomposition and projection are correct. Good dimensionality reduction.",
          executionTime: 80,
        },
      ],
    }

    const assignmentTests = testCases[assignmentId as keyof typeof testCases] || []
    let totalScore = 0
    let maxTotalScore = 0

    for (const testCase of assignmentTests) {
      totalScore += testCase.points
      maxTotalScore += testCase.maxPoints
    }

    const finalScore = Math.floor((totalScore / maxTotalScore) * 100)

    const codeQuality = {
      score: Math.floor(Math.random() * 20) + 80, // 80-100
      issues: finalScore < 85 ? ["Consider adding more comments", "Some functions could be optimized"] : [],
      suggestions: ["Add unit tests", "Consider using type hints", "Add logging for debugging"],
    }

    let overallFeedback = ""
    if (finalScore >= 90) {
      overallFeedback = "Outstanding implementation! Your code demonstrates excellent understanding of ML algorithms."
    } else if (finalScore >= 80) {
      overallFeedback = "Great work! Your implementation covers most requirements with good coding practices."
    } else if (finalScore >= 70) {
      overallFeedback = "Good effort! Focus on completing the remaining algorithm implementations."
    } else {
      overallFeedback = "Keep working on it! Review the requirements and implement the missing components."
    }

    return {
      score: finalScore,
      maxScore: 100,
      testResults: assignmentTests,
      codeQuality,
      overallFeedback,
      gradedAt: new Date().toLocaleString(),
    }
  }

  const submitAssignment = async () => {
    const files = uploadedFiles[activeAssignment] || []

    if (files.length === 0) return

    setAssignmentStates((prev) => ({
      ...prev,
      [activeAssignment]: {
        ...prev[activeAssignment],
        status: "in_progress",
        completionPercentage: 50,
      },
    }))

    try {
      const gradeResult = await runAutoGrader(files, activeAssignment)
      setAutoGradeResults((prev) => ({
        ...prev,
        [activeAssignment]: gradeResult,
      }))

      const mcqScore = calculateMCQScore()
      const feedback = `Assignment submitted successfully! Code score: ${gradeResult.score}%, MCQ score: ${mcqScore.toFixed(1)}%. ${gradeResult.overallFeedback}`

      setAssignmentStates((prev) => ({
        ...prev,
        [activeAssignment]: {
          ...prev[activeAssignment],
          status: "submitted",
          submissionFeedback: feedback,
          submissionDate: new Date().toLocaleDateString(),
          submittedFiles: files.map((f) => f.name),
          completionPercentage: 100,
        },
      }))
    } catch (error) {
      console.error("Auto-grading failed:", error)
    }
  }

  const submitPeerReview = () => {
    if (newReview.trim()) {
      console.log("Submitting review:", { rating: reviewRating, comment: newReview })
      setNewReview("")
      setReviewRating(5)
    }
  }

  const downloadDataset = (url: string, filename: string) => {
    window.open(url, "_blank")
  }

  const downloadTemplate = () => {
    const blob = new Blob([currentAssignment.template], { type: "text/plain" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `${currentAssignment.id}_template.py`
    a.click()
    URL.revokeObjectURL(url)
  }

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case "Beginner":
        return "bg-green-100 text-green-800"
      case "Intermediate":
        return "bg-yellow-100 text-yellow-800"
      case "Advanced":
        return "bg-red-100 text-red-800"
      default:
        return "bg-gray-100 text-gray-800"
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case "not_started":
        return "bg-gray-100 text-gray-800"
      case "in_progress":
        return "bg-blue-100 text-blue-800"
      case "submitted":
        return "bg-purple-100 text-purple-800"
      case "completed":
        return "bg-green-100 text-green-800"
      default:
        return "bg-gray-100 text-gray-800"
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-50 to-blue-50 p-6 rounded-lg border">
        <div className="flex justify-between items-start mb-4">
          <div>
            <h3 className="text-2xl font-bold mb-2 flex items-center gap-2">
              <Brain className="h-8 w-8 text-purple-600" />
              ML Model Development Assignments
            </h3>
            <p className="text-gray-700 mb-3">
              Master machine learning by implementing algorithms from scratch! Complete 3 comprehensive assignments
              covering binary classification, regression, and multi-class classification using real Kaggle datasets.
            </p>
            <div className="flex items-center gap-4 text-sm">
              <div className="flex items-center gap-1">
                <Clock className="h-4 w-4 text-gray-600" />
                <span>10-13 hours total</span>
              </div>
              <div className="flex items-center gap-1">
                <Star className="h-4 w-4 text-yellow-600" />
                <span>650 points total</span>
              </div>
              <div className="flex items-center gap-1">
                <Trophy className="h-4 w-4 text-purple-600" />
                <span>3 assignments</span>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-center">
              <div className="flex items-center gap-1 text-orange-600">
                <Flame className="h-5 w-5" />
                <span className="font-bold text-lg">0</span>
              </div>
              <span className="text-xs text-gray-600">Day Streak</span>
            </div>
            <div className="text-center">
              <div className="flex items-center gap-1 text-blue-600">
                <Star className="h-5 w-5" />
                <span className="font-bold text-lg">0</span>
              </div>
              <span className="text-xs text-gray-600">Total Points</span>
            </div>
          </div>
        </div>
        <div className="flex gap-2 flex-wrap">
          <Badge variant="outline">Real Kaggle Datasets</Badge>
          <Badge variant="outline">From Scratch Implementation</Badge>
          <Badge variant="outline">Auto-Graded</Badge>
          <Badge variant="outline">MCQ Questions</Badge>
          <Badge variant="outline">Peer Reviews</Badge>
          <Badge variant="outline">Code Examples</Badge>
        </div>
      </div>

      <div className="grid lg:grid-cols-4 gap-6">
        {/* Assignment List */}
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="h-5 w-5" />
                Assignments
              </CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              <div className="space-y-2 p-4">
                {assignments.map((assignment) => {
                  const isActive = activeAssignment === assignment.id
                  const state = assignmentStates[assignment.id]
                  const Icon = assignment.icon

                  return (
                    <Button
                      key={assignment.id}
                      variant={isActive ? "default" : "ghost"}
                      className="w-full justify-start text-left h-auto p-3"
                      onClick={() => setActiveAssignment(assignment.id)}
                    >
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <Icon className="h-4 w-4" />
                          <span className="font-medium text-sm">{assignment.title.split(":")[0]}</span>
                        </div>
                        <div className="text-xs text-gray-600 mb-2">{assignment.category}</div>
                        <div className="flex items-center gap-2 mb-2">
                          <Badge variant="secondary" className={`text-xs ${getDifficultyColor(assignment.difficulty)}`}>
                            {assignment.difficulty}
                          </Badge>
                          <Badge variant="secondary" className={`text-xs ${getStatusColor(state.status)}`}>
                            {state.status.replace("_", " ")}
                          </Badge>
                        </div>
                        {state.completionPercentage > 0 && (
                          <div className="mb-2">
                            <Progress value={state.completionPercentage} className="h-2" />
                            <span className="text-xs text-gray-600">{state.completionPercentage}% complete</span>
                          </div>
                        )}
                        <div className="flex items-center gap-4 text-xs text-gray-600">
                          <span className="flex items-center gap-1">
                            <Clock className="h-3 w-3" />
                            {assignment.estimatedTime}
                          </span>
                          <span className="flex items-center gap-1">
                            <Star className="h-3 w-3" />
                            {assignment.points}
                          </span>
                        </div>
                      </div>
                    </Button>
                  )
                })}
              </div>
            </CardContent>
          </Card>

          {/* Achievements */}
          <Card className="mt-4">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Trophy className="h-5 w-5" />
                Achievements
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {achievements.map((achievement) => (
                  <div
                    key={achievement.id}
                    className={`flex items-center gap-3 p-2 rounded ${
                      achievement.unlocked ? "bg-green-50" : "bg-gray-50"
                    }`}
                  >
                    <span className="text-lg">{achievement.icon}</span>
                    <div className="flex-1">
                      <div className="font-medium text-sm">{achievement.name}</div>
                      <div className="text-xs text-gray-600">{achievement.description}</div>
                    </div>
                    {achievement.unlocked && <CheckCircle className="h-4 w-4 text-green-600" />}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Assignment Content */}
        <div className="lg:col-span-3">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-xl flex items-center gap-2">
                    <currentAssignment.icon className="h-6 w-6" />
                    {currentAssignment.title}
                  </CardTitle>
                  <CardDescription className="text-base mt-2">{currentAssignment.description}</CardDescription>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant="outline" className="flex items-center gap-1">
                    <Star className="h-3 w-3" />
                    {currentAssignment.points} pts
                  </Badge>
                  <Badge variant="outline">Due: {currentAssignment.dueDate}</Badge>
                </div>
              </div>

              {/* Progress Indicator */}
              {currentState.completionPercentage > 0 && (
                <div className="mt-4">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium">Progress</span>
                    <span className="text-sm text-gray-600">{currentState.completionPercentage}%</span>
                  </div>
                  <Progress value={currentState.completionPercentage} className="h-3" />
                </div>
              )}
            </CardHeader>

            <CardContent>
              <Tabs defaultValue="overview" className="w-full">
                <TabsList className="grid w-full grid-cols-7">
                  <TabsTrigger value="overview">Overview</TabsTrigger>
                  <TabsTrigger value="datasets">Datasets</TabsTrigger>
                  <TabsTrigger value="examples">Examples</TabsTrigger>
                  <TabsTrigger value="mcqs">MCQs</TabsTrigger>
                  <TabsTrigger value="hints">Hints</TabsTrigger>
                  <TabsTrigger value="reviews">Reviews</TabsTrigger>
                  <TabsTrigger value="submit">Submit</TabsTrigger>
                </TabsList>

                <TabsContent value="overview" className="space-y-4">
                  <div className="grid md:grid-cols-2 gap-6">
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-lg flex items-center gap-2">
                          <Target className="h-5 w-5 text-green-600" />
                          Learning Objectives
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <ul className="space-y-2 text-sm">
                          {currentAssignment.learningObjectives.map((objective, index) => (
                            <li key={index} className="flex items-start gap-2">
                              <CheckCircle className="h-4 w-4 text-green-600 mt-0.5 flex-shrink-0" />
                              <span>{objective}</span>
                            </li>
                          ))}
                        </ul>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader>
                        <CardTitle className="text-lg flex items-center gap-2">
                          <Download className="h-5 w-5 text-purple-600" />
                          Template Code
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <p className="text-sm text-gray-600 mb-3">
                          Download the complete starter template with TODO comments:
                        </p>
                        <Button onClick={downloadTemplate} className="w-full">
                          <Download className="h-4 w-4 mr-2" />
                          Download Template (.py)
                        </Button>
                      </CardContent>
                    </Card>
                  </div>

                  {/* Template Preview */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-lg">📝 Template Preview</CardTitle>
                      <CardDescription>Preview of the starter code you'll be working with locally.</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm max-h-96 overflow-y-auto">
                        <pre className="whitespace-pre-wrap">{currentAssignment.template.slice(0, 1500)}...</pre>
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>

                <TabsContent value="datasets" className="space-y-4">
                  <div className="flex justify-between items-center">
                    <h4 className="font-semibold flex items-center gap-2">
                      <FolderDown className="h-5 w-5" />
                      Real Kaggle Datasets
                    </h4>
                    <Badge variant="outline">{currentAssignment.datasets.length} datasets available</Badge>
                  </div>

                  <div className="grid gap-4">
                    {currentAssignment.datasets.map((dataset, index) => (
                      <Card key={index} className="hover:shadow-md transition-shadow">
                        <CardContent className="p-4">
                          <div className="flex justify-between items-start mb-3">
                            <div className="flex-1">
                              <h5 className="font-medium text-lg mb-1">{dataset.name}</h5>
                              <p className="text-sm text-gray-600 mb-2">{dataset.description}</p>
                              <div className="flex gap-2 mb-2">
                                <Badge variant="outline" className="text-xs">
                                  {dataset.format}
                                </Badge>
                                <Badge variant="secondary" className="text-xs">
                                  {dataset.size}
                                </Badge>
                                {dataset.samples && (
                                  <Badge variant="secondary" className="text-xs">
                                    {dataset.samples.toLocaleString()} samples
                                  </Badge>
                                )}
                              </div>
                              {dataset.features && (
                                <div className="text-xs text-gray-500 mb-2">
                                  <strong>Features:</strong> {dataset.features.slice(0, 4).join(", ")}
                                  {dataset.features.length > 4 && ` +${dataset.features.length - 4} more`}
                                </div>
                              )}
                              {dataset.target && (
                                <div className="text-xs text-gray-500">
                                  <strong>Target:</strong> {dataset.target}
                                </div>
                              )}
                            </div>
                          </div>
                          <div className="flex gap-2">
                            <Button
                              variant="default"
                              size="sm"
                              className="flex-1"
                              onClick={() => downloadDataset(dataset.url, dataset.filename)}
                            >
                              <Download className="h-4 w-4 mr-2" />
                              Download {dataset.filename}
                            </Button>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => window.open(dataset.kaggleUrl, "_blank")}
                            >
                              <ExternalLink className="h-4 w-4" />
                              Kaggle
                            </Button>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>

                  {/* Setup Instructions */}
                  <Card className="bg-blue-50 border-blue-200">
                    <CardHeader>
                      <CardTitle className="text-lg text-blue-800 flex items-center gap-2">
                        <Laptop className="h-5 w-5" />
                        Setup Instructions
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <ol className="text-sm text-blue-800 space-y-2">
                        <li>1. Create a new folder for this assignment</li>
                        <li>2. Download all datasets from Kaggle (may require free account)</li>
                        <li>3. Download the template code and save in the same folder</li>
                        <li>4. Install required packages: pandas, numpy, matplotlib, seaborn</li>
                        <li>5. Open in your Python environment and start coding!</li>
                      </ol>
                    </CardContent>
                  </Card>
                </TabsContent>

                <TabsContent value="examples" className="space-y-4">
                  <h4 className="font-semibold flex items-center gap-2">
                    <Code className="h-5 w-5" />
                    Code Examples & Implementation Guide
                  </h4>

                  <Card>
                    <CardHeader>
                      <CardTitle className="text-lg">💡 Implementation Example</CardTitle>
                      <CardDescription>
                        Study this example to understand the implementation approach for{" "}
                        {currentAssignment.category.toLowerCase()}.
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm max-h-96 overflow-y-auto">
                        <pre className="whitespace-pre-wrap">{currentAssignment.codeExample}</pre>
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle className="text-lg">🔧 Key Implementation Tips</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3">
                        {currentAssignment.id === "titanic-classification" && (
                          <div className="space-y-2">
                            <div className="p-3 bg-blue-50 rounded-lg">
                              <h5 className="font-semibold text-blue-900">Logistic Regression Tips:</h5>
                              <ul className="text-sm text-blue-800 mt-1 space-y-1">
                                <li>• Use gradient descent with learning rate around 0.01</li>
                                <li>• Clip z values in sigmoid to prevent overflow</li>
                                <li>• Monitor cost function to ensure convergence</li>
                                <li>• Initialize weights to zeros, not random values</li>
                              </ul>
                            </div>
                            <div className="p-3 bg-green-50 rounded-lg">
                              <h5 className="font-semibold text-green-900">Feature Engineering Ideas:</h5>
                              <ul className="text-sm text-green-800 mt-1 space-y-1">
                                <li>• Extract titles from names (Mr, Mrs, Miss, Master)</li>
                                <li>• Create FamilySize = SibSp + Parch + 1</li>
                                <li>• Create IsAlone binary feature</li>
                                <li>• Group ages into categories (Child, Adult, Senior)</li>
                              </ul>
                            </div>
                          </div>
                        )}

                        {currentAssignment.id === "house-price-regression" && (
                          <div className="space-y-2">
                            <div className="p-3 bg-purple-50 rounded-lg">
                              <h5 className="font-semibold text-purple-900">Regularization Tips:</h5>
                              <ul className="text-sm text-purple-800 mt-1 space-y-1">
                                <li>• Ridge: Use α values like [0.1, 1, 10, 100] with cross-validation</li>
                                <li>• Lasso: Implement coordinate descent with soft thresholding</li>
                                <li>• Always scale features before applying regularization</li>
                                <li>• Don't regularize the bias term</li>
                              </ul>
                            </div>
                            <div className="p-3 bg-orange-50 rounded-lg">
                              <h5 className="font-semibold text-orange-900">Feature Engineering:</h5>
                              <ul className="text-sm text-orange-800 mt-1 space-y-1">
                                <li>• Create TotalSF = 1stFlrSF + 2ndFlrSF + TotalBsmtSF</li>
                                <li>• Calculate house age = YrSold - YearBuilt</li>
                                <li>• Combine quality scores: OverallQual × OverallCond</li>
                                <li>• Log-transform skewed features like SalePrice</li>
                              </ul>
                            </div>
                          </div>
                        )}

                        {currentAssignment.id === "iris-multiclass" && (
                          <div className="space-y-2">
                            <div className="p-3 bg-pink-50 rounded-lg">
                              <h5 className="font-semibold text-pink-900">KNN Implementation:</h5>
                              <ul className="text-sm text-pink-800 mt-1 space-y-1">
                                <li>• Try different k values (3, 5, 7) and use cross-validation</li>
                                <li>• Implement both Euclidean and Manhattan distance</li>
                                <li>• Use majority voting for class prediction</li>
                                <li>• Consider feature scaling for distance-based algorithms</li>
                              </ul>
                            </div>
                            <div className="p-3 bg-teal-50 rounded-lg">
                              <h5 className="font-semibold text-teal-900">Neural Network Tips:</h5>
                              <ul className="text-sm text-teal-800 mt-1 space-y-1">
                                <li>• Use ReLU activation in hidden layer, softmax in output</li>
                                <li>• Initialize weights with small random values</li>
                                <li>• Implement backpropagation with chain rule</li>
                                <li>• Start with learning rate 0.01 and adjust if needed</li>
                              </ul>
                            </div>
                          </div>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>

                <TabsContent value="mcqs" className="space-y-4">
                  <div className="flex justify-between items-center">
                    <h4 className="font-semibold flex items-center gap-2">
                      <HelpCircle className="h-5 w-5" />
                      Multiple Choice Questions
                    </h4>
                    <Badge variant="outline">
                      {Object.keys(currentState.mcqAnswers).length}/{currentAssignment.mcqs.length} answered
                    </Badge>
                  </div>
                  <div className="space-y-4">
                    {currentAssignment.mcqs.map((mcq, index) => (
                      <Card key={mcq.id}>
                        <CardHeader>
                          <CardTitle className="text-base">
                            Question {index + 1}: {mcq.question}
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-2">
                            {mcq.options.map((option, optionIndex) => (
                              <label
                                key={optionIndex}
                                className={`flex items-center space-x-3 p-3 rounded-lg border cursor-pointer transition-colors ${
                                  currentState.mcqAnswers[mcq.id] === optionIndex
                                    ? "bg-blue-50 border-blue-300"
                                    : "hover:bg-gray-50"
                                }`}
                              >
                                <input
                                  type="radio"
                                  name={`mcq-${mcq.id}`}
                                  value={optionIndex}
                                  checked={currentState.mcqAnswers[mcq.id] === optionIndex}
                                  onChange={() => handleMCQAnswer(mcq.id, optionIndex)}
                                  className="text-blue-600"
                                />
                                <span className="text-sm">{option}</span>
                              </label>
                            ))}
                          </div>

                          {currentState.mcqAnswers[mcq.id] !== undefined && (
                            <div className="mt-4 p-3 bg-gray-50 rounded-lg">
                              <div className="flex items-center gap-2 mb-2">
                                {currentState.mcqAnswers[mcq.id] === mcq.correct ? (
                                  <CheckCircle className="h-4 w-4 text-green-600" />
                                ) : (
                                  <AlertCircle className="h-4 w-4 text-red-600" />
                                )}
                                <span className="font-medium text-sm">
                                  {currentState.mcqAnswers[mcq.id] === mcq.correct ? "Correct!" : "Incorrect"}
                                </span>
                              </div>
                              <p className="text-sm text-gray-700">{mcq.explanation}</p>
                            </div>
                          )}
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                  {Object.keys(currentState.mcqAnswers).length === currentAssignment.mcqs.length && (
                    <Card className="bg-blue-50 border-blue-200">
                      <CardContent className="p-4">
                        <div className="text-center">
                          <Trophy className="h-8 w-8 text-blue-600 mx-auto mb-2" />
                          <h4 className="font-semibold text-blue-900">
                            MCQ Score: {currentState.mcqScore.toFixed(1)}%
                          </h4>
                          <p className="text-sm text-blue-800">
                            You got{" "}
                            {
                              Object.values(currentState.mcqAnswers).filter(
                                (answer, index) => answer === currentAssignment.mcqs[index].correct,
                              ).length
                            }{" "}
                            out of {currentAssignment.mcqs.length} questions correct!
                          </p>
                        </div>
                      </CardContent>
                    </Card>
                  )}
                  ## ❓ **MCQ Questions (5 per assignment):** - **Conceptual questions** about algorithm theory and
                  implementation - **Dataset-specific questions** about feature engineering and preprocessing -
                  **Mathematical questions** about formulas and calculations - **Practical questions** about model
                  evaluation and selection - **Immediate feedback** with detailed explanations for each answer ## 🎯
                  **Key Features:** - **Progressive hint system** - unlock hints as you progress - **Auto-grading
                  system** - immediate feedback on code submissions - **Peer review system** - learn from classmates'
                  approaches - **Achievement badges** - gamification to motivate learning - **Real-time leaderboard** -
                  compete with classmates - **Template code** - structured starting point for each assignment - **Video
                  tutorials** - visual explanations of complex concepts ## 🚀 **Learning Path:** 1. **Start with
                  Titanic** (Binary Classification) - Learn the fundamentals 2. **Progress to House Prices**
                  (Regression) - Master advanced techniques 3. **Complete with Iris** (Multi-class) - Explore neural
                  networks and KNN Each assignment builds upon the previous one, creating a comprehensive learning
                  experience that covers all major ML algorithm categories!
                  <Actions>
                    <Action
                      name="Start Titanic Assignment"
                      description="Begin with binary classification using the Titanic dataset"
                    />
                    <Action
                      name="Download Kaggle Datasets"
                      description="Get real competition data for hands-on practice"
                    />
                    <Action
                      name="View Code Examples"
                      description="Study implementation examples for each algorithm type"
                    />
                    <Action
                      name="Take Practice MCQs"
                      description="Test your understanding with interactive questions"
                    />
                    <Action name="Join Study Group" description="Collaborate with classmates on assignments" />
                  </Actions>
                </TabsContent>

                <TabsContent value="hints" className="space-y-4">
                  <div className="flex justify-between items-center">
                    <h4 className="font-semibold">💡 Progressive Hints</h4>
                    <Badge variant="outline">
                      {currentState.unlockedHints}/{currentAssignment.hints.length} unlocked
                    </Badge>
                  </div>

                  <div className="space-y-3">
                    {currentAssignment.hints.map((hint, index) => {
                      const isUnlocked = index < currentState.unlockedHints
                      return (
                        <Card key={hint.id} className={isUnlocked ? "" : "opacity-50"}>
                          <CardHeader className="pb-3">
                            <div className="flex items-center justify-between">
                              <CardTitle className="text-base flex items-center gap-2">
                                <Lightbulb className={`h-4 w-4 ${isUnlocked ? "text-yellow-500" : "text-gray-400"}`} />
                                Hint {hint.id}: {hint.title}
                              </CardTitle>
                              {isUnlocked && (
                                <Badge variant="secondary" className="text-xs">
                                  Unlocked
                                </Badge>
                              )}
                            </div>
                          </CardHeader>
                          <CardContent className="pt-0">
                            {isUnlocked ? (
                              <p className="text-sm text-gray-700">{hint.content}</p>
                            ) : (
                              <div className="flex items-center gap-2 text-sm text-gray-500">
                                <AlertCircle className="h-4 w-4" />
                                <span>Complete more tasks to unlock this hint</span>
                              </div>
                            )}
                          </CardContent>
                        </Card>
                      )
                    })}
                  </div>

                  {currentState.unlockedHints < currentAssignment.hints.length && (
                    <Button onClick={unlockNextHint} variant="outline" className="w-full bg-transparent">
                      <Lightbulb className="h-4 w-4 mr-2" />
                      Unlock Next Hint
                    </Button>
                  )}
                </TabsContent>

                <TabsContent value="reviews" className="space-y-4">
                  <div className="flex justify-between items-center">
                    <h4 className="font-semibold flex items-center gap-2">
                      <MessageSquare className="h-5 w-5" />
                      Peer Reviews
                    </h4>
                    <Badge variant="outline">
                      {peerReviews.filter((r) => r.assignmentId === activeAssignment).length} reviews
                    </Badge>
                  </div>

                  {/* Submit Review */}
                  <Card className="bg-blue-50 border-blue-200">
                    <CardHeader>
                      <CardTitle className="text-lg">Submit Your Review</CardTitle>
                      <CardDescription>Help your classmates improve their ML implementations</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        <div>
                          <label className="text-sm font-medium mb-2 block">Rating</label>
                          <div className="flex gap-1">
                            {[1, 2, 3, 4, 5].map((star) => (
                              <Button
                                key={star}
                                variant="ghost"
                                size="sm"
                                className="p-1"
                                onClick={() => setReviewRating(star)}
                              >
                                <Star
                                  className={`h-5 w-5 ${
                                    star <= reviewRating ? "fill-yellow-400 text-yellow-400" : "text-gray-300"
                                  }`}
                                />
                              </Button>
                            ))}
                          </div>
                        </div>
                        <div>
                          <label className="text-sm font-medium mb-2 block">Review Comment</label>
                          <textarea
                            value={newReview}
                            onChange={(e) => setNewReview(e.target.value)}
                            placeholder="Share constructive feedback about the algorithm implementation, code quality, or approach..."
                            className="w-full h-24 p-3 border rounded-lg resize-none"
                          />
                        </div>
                        <Button onClick={submitPeerReview} disabled={!newReview.trim()}>
                          <Send className="h-4 w-4 mr-2" />
                          Submit Review
                        </Button>
                      </div>
                    </CardContent>
                  </Card>

                  {/* Existing Reviews */}
                  <div className="space-y-4">
                    {peerReviews
                      .filter((review) => review.assignmentId === activeAssignment)
                      .map((review) => (
                        <Card key={review.id}>
                          <CardContent className="p-4">
                            <div className="flex items-start gap-3">
                              <img
                                src={review.avatar || "/placeholder.svg"}
                                alt={review.reviewer}
                                className="w-8 h-8 rounded-full"
                              />
                              <div className="flex-1">
                                <div className="flex items-center gap-2 mb-2">
                                  <span className="font-medium">{review.reviewer}</span>
                                  <div className="flex gap-1">
                                    {[1, 2, 3, 4, 5].map((star) => (
                                      <Star
                                        key={star}
                                        className={`h-3 w-3 ${
                                          star <= review.rating ? "fill-yellow-400 text-yellow-400" : "text-gray-300"
                                        }`}
                                      />
                                    ))}
                                  </div>
                                  <span className="text-xs text-gray-500">{review.timestamp}</span>
                                </div>
                                <p className="text-sm text-gray-700 mb-3">{review.comment}</p>
                                <div className="flex items-center gap-4">
                                  <Button variant="ghost" size="sm" className="text-xs">
                                    <ThumbsUp className="h-3 w-3 mr-1" />
                                    Helpful ({review.helpful})
                                  </Button>
                                  <Button variant="ghost" size="sm" className="text-xs">
                                    <ThumbsDown className="h-3 w-3 mr-1" />
                                    Not helpful
                                  </Button>
                                  <Button variant="ghost" size="sm" className="text-xs">
                                    <Flag className="h-3 w-3 mr-1" />
                                    Report
                                  </Button>
                                </div>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                  </div>
                </TabsContent>

                <TabsContent value="submit" className="space-y-4">
                  <h4 className="font-semibold">📤 Assignment Submission</h4>

                  {/* File Upload */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-lg flex items-center gap-2">
                        <FileUp className="h-5 w-5 text-blue-600" />
                        Upload Your Files
                      </CardTitle>
                      <CardDescription>
                        Upload your completed Python files (.py, .ipynb) and any additional materials
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-blue-400 transition-colors">
                          <input
                            type="file"
                            multiple
                            accept=".py,.ipynb,.png,.jpg,.jpeg,.pdf,.txt,.csv"
                            onChange={handleFileUpload}
                            className="hidden"
                            id="file-upload"
                          />
                          <label htmlFor="file-upload" className="cursor-pointer">
                            <Upload className="h-12 w-12 mx-auto mb-4 text-gray-400" />
                            <p className="text-lg font-medium mb-2">Drop files here or click to upload</p>
                            <p className="text-sm text-gray-600">
                              Accepted formats: .py, .ipynb, .png, .jpg, .pdf, .txt, .csv
                            </p>
                          </label>
                        </div>

                        {/* Uploaded Files */}
                        {uploadedFiles[activeAssignment] && uploadedFiles[activeAssignment].length > 0 && (
                          <div className="space-y-2">
                            <h5 className="font-medium">Uploaded Files:</h5>
                            {uploadedFiles[activeAssignment].map((file, index) => (
                              <div
                                key={index}
                                className="flex items-center gap-3 p-3 bg-green-50 border border-green-200 rounded-lg"
                              >
                                <CheckCircle className="h-4 w-4 text-green-600" />
                                <span className="flex-1 text-sm">{file.name}</span>
                                <Badge variant="secondary" className="text-xs">
                                  {(file.size / 1024).toFixed(1)} KB
                                </Badge>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    </CardContent>
                  </Card>

                  {/* Submission Actions */}
                  <div className="flex gap-4">
                    <Button
                      onClick={submitAssignment}
                      disabled={
                        currentState.status === "submitted" ||
                        !uploadedFiles[activeAssignment]?.length ||
                        currentState.status === "in_progress"
                      }
                      className="flex-1"
                    >
                      {currentState.status === "submitted" ? (
                        <>
                          <Trophy className="h-4 w-4 mr-2" />
                          Submitted Successfully
                        </>
                      ) : currentState.status === "in_progress" ? (
                        <>
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                          Auto-Grading in Progress...
                        </>
                      ) : (
                        <>
                          <Upload className="h-4 w-4 mr-2" />
                          Submit Assignment
                        </>
                      )}
                    </Button>
                    <Button variant="outline">
                      <GitBranch className="h-4 w-4 mr-2" />
                      Save Draft
                    </Button>
                  </div>

                  {/* Submission Feedback */}
                  {currentState.status === "submitted" && (
                    <Card className="bg-green-50 border-green-200">
                      <CardHeader>
                        <CardTitle className="text-lg text-green-900 flex items-center gap-2">
                          <Trophy className="h-5 w-5" />
                          Assignment Submitted Successfully!
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-3">
                          <p className="text-green-800 text-sm">
                            <strong>Submitted on:</strong> {currentState.submissionDate}
                          </p>
                          <p className="text-green-800 text-sm">
                            <strong>Files submitted:</strong> {currentState.submittedFiles.join(", ")}
                          </p>
                          <div className="bg-white p-3 rounded border">
                            <p className="text-sm font-medium mb-2">Feedback:</p>
                            <p className="text-sm">{currentState.submissionFeedback}</p>
                          </div>
                          <div className="flex items-center gap-2 text-sm text-green-700">
                            <Star className="h-4 w-4" />
                            <span>+{currentAssignment.points} points earned!</span>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  )}

                  {/* Auto-Grading Results */}
                  {currentState.status === "submitted" && autoGradeResults[activeAssignment] && (
                    <Card className="bg-gradient-to-r from-blue-50 to-purple-50 border-blue-200">
                      <CardHeader>
                        <CardTitle className="text-lg text-blue-900 flex items-center gap-2">
                          <Trophy className="h-5 w-5" />
                          Auto-Grading Results
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-4">
                          {/* Overall Score */}
                          <div className="flex items-center justify-between p-4 bg-white rounded-lg border">
                            <div>
                              <div className="text-3xl font-bold text-blue-600">
                                {autoGradeResults[activeAssignment].score}/100
                              </div>
                              <div className="text-sm text-gray-600">Code Score</div>
                            </div>
                            <div className="text-right">
                              <div className="text-xl font-semibold">
                                {autoGradeResults[activeAssignment].score >= 90
                                  ? "🏆 Outstanding"
                                  : autoGradeResults[activeAssignment].score >= 80
                                    ? "🥈 Excellent"
                                    : autoGradeResults[activeAssignment].score >= 70
                                      ? "🥉 Good"
                                      : "📚 Needs Work"}
                              </div>
                              <div className="text-xs text-gray-500">
                                MCQ Score: {currentState.mcqScore.toFixed(1)}%
                              </div>
                            </div>
                          </div>

                          {/* Test Results */}
                          <div>
                            <h5 className="font-semibold mb-3 flex items-center gap-2">
                              <CheckCircle className="h-4 w-4" />
                              Detailed Test Results
                            </h5>
                            <div className="space-y-2">
                              {autoGradeResults[activeAssignment].testResults.map((test, index) => (
                                <div
                                  key={index}
                                  className="flex items-center justify-between p-3 bg-white rounded border"
                                >
                                  <div className="flex items-center gap-3">
                                    {test.passed ? (
                                      <CheckCircle className="h-4 w-4 text-green-500" />
                                    ) : (
                                      <AlertCircle className="h-4 w-4 text-red-500" />
                                    )}
                                    <div>
                                      <div className="font-medium text-sm">{test.testName}</div>
                                      <div className="text-xs text-gray-600">{test.feedback}</div>
                                    </div>
                                  </div>
                                  <div className="text-right">
                                    <div className="font-semibold text-sm">
                                      {test.points}/{test.maxPoints}
                                    </div>
                                    {test.executionTime && (
                                      <div className="text-xs text-gray-500">{test.executionTime}ms</div>
                                    )}
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>

                          {/* Overall Feedback */}
                          <div className="p-3 bg-white rounded border">
                            <div className="text-sm font-medium mb-2">Overall Feedback:</div>
                            <p className="text-sm text-gray-700">
                              {autoGradeResults[activeAssignment].overallFeedback}
                            </p>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  )}

                  {/* Leaderboard */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-lg flex items-center gap-2">
                        <Crown className="h-5 w-5 text-yellow-500" />
                        Class Leaderboard
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3">
                        {leaderboard.map((student) => (
                          <div
                            key={student.rank}
                            className={`flex items-center gap-3 p-3 rounded-lg ${
                              student.name === "You" ? "bg-blue-50 border border-blue-200" : "bg-gray-50"
                            }`}
                          >
                            <div className="flex items-center gap-2">
                              {student.rank === 1 && <Crown className="h-4 w-4 text-yellow-500" />}
                              {student.rank === 2 && <Medal className="h-4 w-4 text-gray-400" />}
                              {student.rank === 3 && <Award className="h-4 w-4 text-orange-500" />}
                              <span className="font-bold text-lg">{student.rank}</span>
                            </div>
                            <img
                              src={student.avatar || "/placeholder.svg"}
                              alt={student.name}
                              className="w-8 h-8 rounded-full"
                            />
                            <div className="flex-1">
                              <div className="font-medium">{student.name}</div>
                              <div className="text-sm text-gray-600">{student.points} points</div>
                            </div>
                            <div className="flex items-center gap-1 text-orange-500">
                              <Flame className="h-4 w-4" />
                              <span className="text-sm font-medium">{student.streak}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
