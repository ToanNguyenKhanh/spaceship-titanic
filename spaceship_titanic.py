import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lazypredict.Supervised import LazyRegressor, LazyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report

"""Import Dataset"""

test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")
# print(test.info())
print(train.info())

missing_train = train.isnull().sum()
missing_test = test.isnull().sum()

print("Percentage Missing train data\n", train.isnull().mean() * 100, )
print("=" * 20)
print("Percentage Missing of test data\n", test.isnull().mean() * 100)
# print(missing_train)
# print(missing_test)


# Vẽ heatmap cho dữ liệu bị thiếu|
# plt.figure(figsize=(12, 8))
# sns.heatmap(train.isnull(), cbar=False, cmap='viridis', yticklabels=False)
# plt.title('Visualization of Missing Data')
# plt.show()

"""# Checking balanced data or not"""

class_counts = train['Transported'].value_counts()
print(class_counts)

plt.figure(figsize=(10, 6))
sns.countplot(data=train, x='Transported')
plt.title('Class Distribution')
plt.show()

sns.heatmap(train.corr(), annot=True)
plt.show()

"""Biểu đồ corr yếu nên khả năng cao xài model tuyến tính sẽ cho kết quả không được tốt"""

target = 'Transported'
x = train.drop([target, "PassengerId", "Name", "Cabin"], axis=1)
y = train[target]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(y_train.shape)
print(X_train.shape)

"""#Pipeline:
- Preprocessing
- Imputer
- Transformer
"""

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', StandardScaler())
])

print((X_train["HomePlanet"].unique()))
print((X_train["Destination"].unique()))
print((X_train["VIP"].unique()))

# HomePlanet_vals = ['Earth','Europa','Mars']
# CryoSleep_vals = X_train["CryoSleep"].unique()
# Destination_vals = X_train["Destination"].unique()
# VIP_vals = X_train["VIP"].unique()

HomePlanet_vals = ['Earth', 'Europa', 'Mars']
CryoSleep_vals = X_train["CryoSleep"].dropna().unique().tolist()
Destination_vals = X_train["Destination"].dropna().unique().tolist()
VIP_vals = X_train["VIP"].dropna().unique().tolist()
print(HomePlanet_vals)
print(CryoSleep_vals)
print(Destination_vals)
print(VIP_vals)

ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="most_frequent")),  # Xử lý missing value
    # ('scaler', OrdinalEncoder(categories=[HomePlanet_vals, X_train["CryoSleep"].unique(), X_train["Destination"].unique(), X_train["VIP"].unique()]))
    ('scaler', OrdinalEncoder(categories=[CryoSleep_vals, VIP_vals]))  # Biến đổi dữ liệu
])

norminal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="most_frequent")),  # Xử lý missing value
    ('scaler', OneHotEncoder())  # Biến đổi dữ liệu
])

"""# Preprocessor"""

preprocessor = ColumnTransformer(transformers=[
    ("num_features", numerical_transformer, ["Age", 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']),
    ('ord_features', ordinal_transformer, ["CryoSleep", 'VIP']),
    ('nom_features', norminal_transformer, ['HomePlanet', 'Destination'])
])

"""#pipeline"""

cls_lazy = Pipeline(steps=[
    ('preprocessor', preprocessor),
    # ('model', RandomForestRegressor())
])

"""# LazyPredict"""

X_train = cls_lazy.fit_transform((X_train))
X_test = cls_lazy.transform(X_test)
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models)

"""# GridSearch"""

# Logistic Regression
# LGBMClassifier
# RandomForestClassifier

cls = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LogisticRegression())
    # ('model', RandomForestClassifier())

])

# Logistic regression parameters
lr_params = {
    "model__penalty": ['l1', 'l2', 'elasticnet', None],
    "model__solver": ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
}

#  RandomForestClassifier parameters
rf_params = {
    "model__criterion": ["gini", "entropy", "log_loss"],
    "model__max_features": ["sqrt", "log2", None],
}

grid_cls = GridSearchCV(cls, param_grid=lr_params, cv=6, verbose=2)
grid_cls.fit(X_train, y_train)
y_predict = grid_cls.predict(X_test)
print(classification_report(y_test, y_predict))

submission_pred = grid_cls.predict(test)

print(submission_pred.shape)

test_ids = test['PassengerId']
print(test_ids.shape)

submission_pred = submission_pred.astype(bool)

df = pd.DataFrame({'PassengerId': test_ids.values, 'Transported': submission_pred})

df.to_csv("/content/drive/MyDrive/Data Science/Kaggle Competition/Spaceship_titanic/Submission.csv", index=False)
