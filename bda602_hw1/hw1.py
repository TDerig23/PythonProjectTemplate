import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# r"/mnt/d/Data/CSV_file.csv"
datafile = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# c=pd.read_csv(datafile)


iris_data = pd.read_csv(datafile, sep=",")
# print(iris_data)
iris_data.columns = [
    "Sepal_Length",
    "Sepal_width",
    "petal_length",
    "petal_width",
    "species",
]

# print(iris_data.head())

print(iris_data.describe())


def petal_func(columnname):
    column = iris_data[columnname]
    statement = "Column {element} statistics"
    print(statement.format(element=columnname))
    print(str(np.mean(column)) + " Mean of column")
    print(str(np.median(column)) + " Median of Column")
    print(str(np.std(column)) + " Standard Deviation of Column")

    return column


petal_func("Sepal_Length")
petal_func("Sepal_width")
petal_func("petal_length")
petal_func("petal_width")


fig = px.histogram(iris_data, x="species", y="Sepal_Length")
fig.show()

fig2 = px.scatter(iris_data, x="species", y="Sepal_Length")
fig2.show()

fig3 = px.pie(iris_data, values="petal_width", names="species")
fig3.show()

fig4 = px.box(iris_data, x="species", y="Sepal_width", points="all")
fig4.show()

fig5 = px.violin(iris_data, x="species", y="petal_length", violinmode="overlay")
fig5.show()

# Analyze and build models - Use scikit-learn
# Use the StandardScaler transformer
# Fit the transformed data against random forest classifier (try other classifiers)

# standard scalar
species = iris_data["species"]
data_new = iris_data.drop(columns=["species"])
xtrain, xtest, ytrain, ytest = train_test_split(data_new, species, test_size=0.2)
sc = StandardScaler()
xtrain_sc = sc.fit_transform(xtrain)
xtest_sc = sc.transform(xtest)
print(xtrain)


# random forest
random_forest = RandomForestClassifier(n_estimators=100, random_state=123)
random_forest.fit(xtrain_sc, ytrain)
random_forest_predict = random_forest.predict(xtrain_sc)
print(random_forest_predict)

# Gaussian Process Classifier
kernel = 1.0 * RBF(1.0)
gpc = GaussianProcessClassifier(kernel=kernel, random_state=0).fit(xtrain_sc, ytrain)

print(gpc.score(xtrain_sc, ytrain))

# adaboost classifier

clf = AdaBoostClassifier(n_estimators=100, random_state=0)
clf.fit(xtrain_sc, ytrain)
clf_predict = clf.predict(xtrain_sc)
print(clf_predict)

# pipeline
pipe = Pipeline(
    [("scaler", StandardScaler()), ("classifier", RandomForestClassifier())]
)
pipe.fit(xtrain_sc, ytrain)
pipe_score = pipe.score(xtest_sc, ytest)
print(f"Pipeline_Score: {pipe_score}")
