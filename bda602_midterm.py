import random
from typing import List
from scipy import stats
import pandas as pd
import seaborn
from sklearn import datasets
import sys

import numpy
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from sklearn.metrics import confusion_matrix

titanic_df = pd.read_csv(
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")


TITANIC_PREDICTORS = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "embarked",
    "parch",
    "fare",
    "who",
    "adult_male",
    "deck",
    "embark_town",
    "alone",
    "class",
]

dataframe = pd.DataFrame()


def get_test_data_set(data_set_name: str = None) -> (pd.DataFrame, List[str], str):
    """Function to load a few test data sets

    :param:
    data_set_name : string, optional
        Data set to load

    :return:
    data_set : :class:`pandas.DataFrame`
        Tabular data, possibly with some preprocessing applied.
    predictors :list[str]
        List of predictor variables
    response: str
        Response variable
    """
    seaborn_data_sets = ["mpg", "tips", "titanic", "titanic_2"]
    sklearn_data_sets = ["boston", "diabetes", "breast_cancer"]
    all_data_sets = seaborn_data_sets + sklearn_data_sets

    if data_set_name is None:
        data_set_name = random.choice(all_data_sets)
    else:
        if data_set_name not in all_data_sets:
            raise Exception(f"Data set choice not valid: {data_set_name}")

    if data_set_name in seaborn_data_sets:
        if data_set_name == "mpg":
            data_set = seaborn.load_dataset(name="mpg").dropna().reset_index()
            predictors = [
                "cylinders",
                "displacement",
                "horsepower",
                "weight",
                "acceleration",
                "origin",
                "name",
            ]
            response = "mpg"
        elif data_set_name == "tips":
            data_set = seaborn.load_dataset(name="tips").dropna().reset_index()
            predictors = [
                "total_bill",
                "sex",
                "smoker",
                "day",
                "time",
                "size",
            ]
            response = "tip"
        elif data_set_name == "titanic":
            data_set = seaborn.load_dataset(name="titanic").dropna()
            predictors = TITANIC_PREDICTORS
            response = "survived"
        elif data_set_name == "titanic_2":
            data_set = seaborn.load_dataset(name="titanic").dropna()
            predictors = TITANIC_PREDICTORS
            response = "alive"
    elif data_set_name in sklearn_data_sets:
        if data_set_name == "boston":
            data = datasets.load_boston()
            data_set = pd.DataFrame(data.data, columns=data.feature_names)
            data_set["CHAS"] = data_set["CHAS"].astype(str)
        elif data_set_name == "diabetes":
            data = datasets.load_diabetes()
            data_set = pd.DataFrame(data.data, columns=data.feature_names)
            data_set["gender"] = ["1" if i > 0 else "0" for i in data_set["sex"]]
        elif data_set_name == "breast_cancer":
            data = datasets.load_breast_cancer()
            data_set = pd.DataFrame(data.data, columns=data.feature_names)

        data_set["target"] = data.target
        predictors = data.feature_names
        response = "target"

    print(f"Data set selected: {data_set_name}")
    return data_set, predictors, response


dataframe, predictors, response = get_test_data_set("breast_cancer")



sibsp = dataframe['worst texture'].tolist()


# dont separate the dataframe, determine if predictors are continious or categroical and make a list from it.
## also use a separate function to determine if response has n.unique == 2 then boolean (use label encoder)
## loop through to find the datatype of each predictor create a list of the types.
def column_sep(dataframe):
    cat_df = pd.DataFrame()
    cont_df = pd.DataFrame()
    for column in dataframe.columns:
        if dataframe[column].dtypes == "bool" or dataframe[column].dtypes == "object" or len(
                pd.unique(dataframe[column])) == 2:
            cat_df[column] = dataframe[column]
        else:
            cont_df[column] = dataframe[column]

    return cat_df, cont_df


cat_df, cont_df = column_sep(dataframe)


# dont separate the dataframe, determine if predictors are continious or categroical and make a list from it.
## also use a separate function to determine if response has n.unique == 2 then boolean (use label encoder)
## loop through to find the datatype of each predictor create a list of the types.
def column_sep(dataframe):
    predictor_dict = {}
    for column in dataframe.columns:
        if dataframe[column].dtypes == "bool" or dataframe[column].dtypes == "object" or len(
                pd.unique(dataframe[column])) == 2:
            predictor_dict[column] = "categorical"
        else:
            predictor_dict[column] = "continuous"

    return predictor_dict


predictors = column_sep(dataframe)

for key, value in predictors.items():
    print

## get datatype and dataname
predictors


def type_chooser(predictors):
    cont_cont_df = pd.DataFrame()
    for key1 in range(len(predictors)):
        for key2 in range(key1, len(predictors)):
            if predictors.get(key1) == "continuous" and predictors.get(key2) == "continuous":
                res = stats.pearsonr(dataframe[key1], dataframe[key2])
                cont_cont_df["Pearsons_R"] = res
            elif predictors.get(key1) == "continuous" and predictors.get(key2) == "categorical":
                res = stats.pearsonr(dataframe[key1], dataframe[key2])
                cont_cont_df["Pearsons_R"] = res


res = type_chooser(predictors)

print(res)

res = stats.pearsonr([1, 2, 3, 4, 5], [10, 9, 2.5, 6, 4])
res

## for pearsons r continious v continious loop on columns for cont_df (double for loop) get pearson correlation. store in dataframe
## enumerate the index
## Continuous / Categorical pairs df.corr[]
# Categorical / Categorical pairs us cat_correlation

## use for loop to determine the graphing types. for loop then if else to determine if response is boolean or not
## then correspond to correct correlation type.


# for predictor1 in cat df



