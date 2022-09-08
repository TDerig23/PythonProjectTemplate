import pandas as pd
import plotly
import numpy as np
import sklearn
import plotly.express as px


#r"/mnt/d/Data/CSV_file.csv"
datafile ="http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
#c=pd.read_csv(datafile)
#datafile2 = "/mnt/C:/Users/thoma/OneDrive/Documents/bda602/hw1/bezdekIris.data"
#print(datafile)

iris_data = pd.read_csv(datafile, sep=",")
#print(iris_data)
iris_data.columns =['Sepal_Length', 'Sepal_width', 'petal_length', 'petal_width','species']

print(iris_data.head())

# def petal_func(columnname):
#     column = iris_data[columnname]
#     statement = "Column {element} statistics"
#     print(statement.format(element=columnname))
#     print(str(np.mean(column)) + " Mean of column")
#     print(str(np.median(column)) + " Median of Column")
#     print(str(np.std(column)) + " Standard Deviation of Column")
#
#     return column
#
#
# petal_func("5.1")
# petal_func("3.5")
# petal_func("1.4")
# petal_func("0.2")



# fig = px.histogram(iris_data, x="species", y = "Sepal_Length")
# fig.show()

# fig2 = px.scatter(iris_data, x="species", y = "Sepal_Length")
# fig2.show()

# fig3 = px.pie(iris_data, values = "petal_width", names ="species")
# fig3.show()
#
# fig4 = px.line(iris_data, x="species", y = "Sepal_width")
# fig4.show()
# #
# fig5 = px.bar(iris_data, x="species", y = "petal_length")
# fig5.show()

# Analyze and build models - Use scikit-learn
# Use the StandardScaler transformer
# Fit the transformed data against random forest classifier (try other classifiers)

# y = species


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
species = iris_data['species']
data_new = iris_data.drop(columns=['species'])
xtrain, xtest, ytrain, ytest = train_test_split(data_new, test_size=0.2)
sc = StandardScaler()
X_train = sc.fit_transform(xtrain)
X_test = sc.transform(xtest)
print(X_train)

# from sklearn.ensemble import RandomForestClassifier
# xtrain, xtest = train_test_split(data_new, test_size=0.2,random_state=123)
#
# xtrain["Train_or_test"] =1
# xtest["Train_or_test"] = 0
#
# concat_train_test = pd.concat([xtrain,xtest],axis=0)
#
# y_concat = concat_train_test.pop('Train_or_test')
# random_forest = RandomForestClassifier(n_estimators=100,random_state=123)
# random_forest.fit(concat_train_test,y_concat)
#
# y_pred = random_forest.predict(concat_train_test)
# from sklearn.metrics import roc_auc_score
# final = roc_auc_score



