# ======================
# 📦 Imports
# ======================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, LinearRegression, ElasticNetCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler




import joblib
import json



# ======================
# 📂 Load Data
# ======================
print("Loading dataset...")
train = pd.read_csv('e:\HW\\train.csv')

# ======================
# 🔧 Handle Missing Data
# ======================
fill_n = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType', 'FireplaceQu',
          'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
          'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
print("Filling missing values for categorical features with 'None'...")
for c in fill_n:
    train[c] = train[c].fillna('None')

fill_zero = ['GarageYrBlt', 'GarageCars', 'GarageArea', 'MasVnrArea']
print("Filling missing values for numerical features with 0...")
for col in fill_zero:
    train[col] = train[col].fillna(0)

# Fill missing values for LotFrontage with the median
print("Filling LotFrontage missing values with median...")
train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])

# ======================
print("Handling missing values completed.")

print("Preprocessing SalePrice...")
# ======================
train['SalePrice'] = np.log1p(train['SalePrice'])  # log(1 + x)



# Drop outliers using domain knowledge
train = train.drop(train[(train['GrLivArea'] > 4500) & (train['SalePrice'] < 300000)].index)
train = train.drop(train[(train['TotalBsmtSF'] > 4000) & (train['SalePrice'] < 300000)].index)
train = train.drop(train[(train['1stFlrSF'] > 4000) & (train['SalePrice'] < 300000)].index)
train = train.drop(train[(train['GarageArea'] > 1200) & (train['SalePrice'] < 300000)].index)
train = train.drop(train[(train['LotFrontage'] > 300) & (train['SalePrice'] < 300000)].index)
train = train.drop(train[(train['MasVnrArea'] > 1400) & (train['SalePrice'] < 300000)].index)
train = train.drop(train[(train['BsmtFinSF1'] > 3000) & (train['SalePrice'] < 300000)].index)
print("Outliers removed based on domain knowledge.")

#========================#===#
#  Feature Encoding   ##
#=======================##
train.drop(['Id'], axis=1, inplace=True)
train = pd.get_dummies(train)
print("Feature encoding completed. Shape of dataset:", train.shape)

# ======================
# Split Features and Target
# ======================
y = train['SalePrice']
X = train.drop('SalePrice', axis=1)
print("Features and target variable separated.")
# ======================
# Train-Test Split (80/20)
# ======================

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
print("Train-test split completed. Train shape:", X_train.shape, "Test shape:", X_test.shape)
# ====================== Scale Features # ======================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled using StandardScaler.")
# Impute if needed 
imputer = SimpleImputer(strategy='mean')
X_train_scaled = imputer.fit_transform(X_train_scaled)
X_test_scaled = imputer.transform(X_test_scaled)
print("Missing values imputed with mean.")
# ======================
#Train LassoCV
# ======================
lasso_cv = LassoCV(alphas=np.logspace(-4, -0.5, 30), cv=5, max_iter=10000)
lasso_cv.fit(X_train_scaled, y_train)

print(f"LassoCV Best alpha: {lasso_cv.alpha_}")

# Evaluate Lasso
y_train_pred_lasso = lasso_cv.predict(X_train_scaled)
y_test_pred_lasso = lasso_cv.predict(X_test_scaled)
print("LassoCV model trained and predictions made.")

rmse_train_lasso = np.sqrt(mean_squared_error(y_train, y_train_pred_lasso))
rmse_test_lasso = np.sqrt(mean_squared_error(y_test, y_test_pred_lasso))
r2_train_lasso = r2_score(y_train, y_train_pred_lasso)
r2_test_lasso = r2_score(y_test, y_test_pred_lasso)
print("LassoCV model evaluation:")


print(f"LassoCV Train RMSE: {rmse_train_lasso:.4f}, R2: {r2_train_lasso:.4f}")
print(f"LassoCV Test  RMSE: {rmse_test_lasso:.4f}, R2: {r2_test_lasso:.4f}")

print("\n" + "="*30 + "\n")

# ======================
#  Train Linear Regression
# ======================
print("Training Linear Regression model...")
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)

y_train_pred_lin = lin_reg.predict(X_train_scaled)
y_test_pred_lin = lin_reg.predict(X_test_scaled)

print("Linear Regression model trained and predictions made.")
rmse_train_lin = np.sqrt(mean_squared_error(y_train, y_train_pred_lin))
rmse_test_lin = np.sqrt(mean_squared_error(y_test, y_test_pred_lin))
print("Linear Regression model evaluation:")

r2_train_lin = r2_score(y_train, y_train_pred_lin)
r2_test_lin = r2_score(y_test, y_test_pred_lin)

print(f"Linear Regression Train RMSE: {rmse_train_lin:.4f}, R2: {r2_train_lin:.4f}")
print(f"Linear Regression Test  RMSE: {rmse_test_lin:.4f}, R2: {r2_test_lin:.4f}")

print("\n" + "="*30 + "\n")

# ======================
# 📐 Train ElasticNetCV
# ======================
#enet = ElasticNetCV(alphas=np.logspace(-4, -0.5, 30), l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=5, max_iter=10000)

#enet.fit(X_train_scaled, y_train)

#y_train_pred_enet = enet.predict(X_train_scaled)
#y_test_pred_enet = enet.predict(X_test_scaled)

#rmse_train_enet = np.sqrt(mean_squared_error(y_train, y_train_pred_enet))
#rmse_test_enet = np.sqrt(mean_squared_error(y_test, y_test_pred_enet))
#r2_train_enet = r2_score(y_train, y_train_pred_enet)
#r2_test_enet = r2_score(y_test, y_test_pred_enet)

#print(f"ElasticNet best alpha: {enet.alpha_}, l1_ratio: {enet.l1_ratio_}")
#print(f"ElasticNet Train RMSE: {rmse_train_enet:.4f}, R2: {r2_train_enet:.4f}")
#print(f"ElasticNet Test  RMSE: {rmse_test_enet:.4f}, R2: {r2_test_enet:.4f}")

# Define a tolerance for accuracy calculation (e.g., 10% of the actual value)
#tolerance = 0.1

# Calculate accuracy-like metric for ElasticNet
#accuracy_train_enet = np.mean(np.abs(y_train_pred_enet - y_train) <= tolerance * y_train) * 100
#accuracy_test_enet = np.mean(np.abs(y_test_pred_enet - y_test) <= tolerance * y_test) * 100

#print(f"ElasticNet Train Accuracy (within {tolerance*100:.0f}% tolerance): {accuracy_train_enet:.2f}%")
#print(f"ElasticNet Test Accuracy (within {tolerance*100:.0f}% tolerance): {accuracy_test_enet:.2f}%")



#######random_forest########

from sklearn.ensemble import RandomForestRegressor


rn=RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

rn.fit(X_train_scaled, y_train)


y_train_pred_rf = rn.predict(X_train_scaled)
y_test_pred_rf = rn.predict(X_test_scaled)
print(f"random forest model trained")
rmse_train_rf = np.sqrt(mean_squared_error(y_train, y_train_pred_rf))
rmse_test_rf = np.sqrt(mean_squared_error(y_test, y_test_pred_rf))
r2_train_rf = r2_score(y_train, y_train_pred_rf)
r2_test_rf = r2_score(y_test, y_test_pred_rf)

print("\nRandom Forest Results:")
print(f"Train RMSE: {rmse_train_rf:.4f}, R2: {r2_train_rf:.4f}")
print(f"Test  RMSE: {rmse_test_rf:.4f}, R2: {r2_test_rf:.4f}")



joblib.dump(lasso_cv, 'model.pkl') #saveing model
joblib.dump(scaler, 'scaler.pkl') #saving sclaer

# Save feature column names for API use
model_columns = X_train.columns.tolist()
with open('columns.json', 'w') as f:
    json.dump(model_columns, f)

def plot_actual_vs_predicted_subset(y_actual, y_pred, model_name="Model", n=100):
    """
    Plot Actual vs Predicted for the first `n` test samples.
    """
    plt.figure(figsize=(12, 6))

    y_actual_np = y_actual.values if isinstance(y_actual, pd.Series) else y_actual
    y_pred_np = y_pred

    # Subset the first `n` points
    y_actual_sub = y_actual_np[:n]
    y_pred_sub = y_pred_np[:n]

    plt.plot(y_actual_sub, label='Actual', color='red', linewidth=2)
    plt.plot(y_pred_sub, label='Predicted', color='green', linestyle='--', linewidth=2)

    plt.xlabel(f'First {n} Test Data Points')
    plt.ylabel('Log(SalePrice)')
    plt.title(f'{model_name} - Actual vs Predicted (First {n})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_actual_vs_predicted_subset(y_test, y_test_pred_rf, model_name="Random Forest", n=75)
plot_actual_vs_predicted_subset(y_test, y_test_pred_lasso, model_name="LassoCV", n=50)

# ======================
# Compare Models
# ======================
if rmse_test_lasso < rmse_test_rf:
    better_model = "LassoCV"
    best_model = lasso_cv
else:
    better_model = "Random Forest"
    best_model = rn

print(f"\nBetter Performing Model: {better_model}")

# ======================
# Export Better Model
# ======================
joblib.dump(best_model, 'model.pkl')  # Save the better model for the API
print(f"{better_model} model exported as 'model.pkl'")




