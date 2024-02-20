import  pandas as pd 
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.metrics import r2_score
import pickle
df = pd.read_csv('dataset/heart_data.csv')
df = df.drop("Unnamed: 0", axis=1)
# sns.lmplot(x='biking', y='heart.disease', data=df)
# sns.lmplot(x='smoking', y='heart.disease', data=df)


# extract X_train, X_test, y_train, y_test 
x_df = df.drop('heart.disease', axis=1)
y_df = df['heart.disease']

X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.1, random_state=0)


model = linear_model.LinearRegression()
model.fit(X_train, y_train)
print(f' model score is : {model.score(X_train, y_train)}')


prediction_test = model.predict(X_test)


r2_score = r2_score(y_test, prediction_test)
print(f'r2_score is : {r2_score}')

pickle.dump(model, open('model/model.pkl', 'wb'))
model = pickle.load(open('model/model.pkl', 'rb'))

print(model.predict([[70.1, 27]]))

# plt.show()