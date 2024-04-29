from sklearn.linear_model import LogisticRegression
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from joblib import dump

data = pd.read_excel("object-distance-master.xlsx")
label_encoder = LabelEncoder()
X = data[['ObjectName','Size','Region']]

X.loc[:, 'ObjectName'] = label_encoder.fit_transform(X['ObjectName'])

y = data['isClose']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logRegModel = LogisticRegression()
logRegModel.fit(X_train,y_train)

test_score = logRegModel.score(X_test,y_test)
print("Accuracy of the model: ",test_score)


dump((logRegModel, label_encoder), "distance_prediction_logReg-model.joblib")


