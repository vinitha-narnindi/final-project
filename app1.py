import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import pickle
col_names = ['age','gender','chest_pain','blood_pressure','serum_cholestoral','fasting_blood_sugar', 'electrocardiographic',
             'max_heart_rate','induced_angina','ST_depression','slope','no_of_vessels','thal','diagnosis']

# read the file
df = pd.read_csv(r'C:\Users\User\Desktop\ml\processed.cleveland.data.csv', names=col_names, header=None, na_values="?")

df.isnull().sum()

df['no_of_vessels'].fillna(df['no_of_vessels'].mode()[0], inplace=True)
df['thal'].fillna(df['thal'].mode()[0], inplace=True)
df.diagnosis=(df.diagnosis!=0).astype(int)
df.diagnosis.value_counts()
X, y = df.iloc[:, :-1], df.iloc[:, -1]
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

#model1
modellog = LogisticRegression()
# Train the model using the training sets and check score
modellog.fit(X_train, y_train)
pred1= modellog.predict(X_test)
joblib.dump(modellog, 'LogisticRegression.pkl')

#model2
#modelsvm = svm.svc()
modelsvm = svm.SVC(kernel='linear', C=1, gamma=1)
modelsvm.fit(X_train, y_train)
pred2= modelsvm.predict(X_test)
joblib.dump(modelsvm, 'SVM.pkl')

#model3
modelgnb = GaussianNB()
modelgnb.fit(X_train, y_train)
pred3 = modelgnb.predict(X_test)
joblib.dump(modellog, 'NaiveBayes.pkl')

#model4
clf_entropy=DecisionTreeClassifier(criterion = "entropy", random_state =0,max_depth=5,min_samples_leaf=5)
clf_entropy.fit(X_train,y_train)
y_pred_en=clf_entropy.predict(X_test)
joblib.dump(clf_entropy, 'decisiontree.pkl')

#model5

# Choose the type of classifier. 
clf = RandomForestClassifier()

# Choose some parameter combinations to try
parameters = {'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8],
              'n_estimators':[110], 'random_state':[2606]
             }

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# Run the grid search
grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
joblib.dump(clf, 'randomforest.pkl')