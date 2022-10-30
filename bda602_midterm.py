import random
from typing import List
from scipy import stats
import pandas as pd
import seaborn
from sklearn import datasets
import sys
import warnings
import numpy
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from sklearn.metrics import confusion_matrix



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


def column_sep(dataframe):
    predictor_name = []
    predictor_type = []
    for column in dataframe.columns:
        if dataframe[column].dtypes == "bool" or dataframe[column].dtypes == "object" or len(
                pd.unique(dataframe[column])) < 5:
            predictor_name.append(column)
            predictor_type.append("categorical")
        else:
            predictor_name.append(column)
            predictor_type.append("continuous")

    predictor_list = list(map(list, zip(predictor_name, predictor_type)))

    return predictor_list


predictor_list = column_sep(dataframe)

def fill_na(data):
    if isinstance(data, pd.Series):
        return data.fillna(0)
    else:
        return numpy.array([value if value is not None else 0 for value in data])

def cat_correlation(x, y, bias_correction=True, tschuprow=False):
    """
    Calculates correlation statistic for categorical-categorical association.
    The two measures supported are:
    1. Cramer'V ( default )
    2. Tschuprow'T

    SOURCES:
    1.) CODE: https://github.com/MavericksDS/pycorr
    2.) Used logic from:
        https://stackoverflow.com/questions/20892799/using-pandas-calculate-cram%C3%A9rs-coefficient-matrix
        to ignore yates correction factor on 2x2
    3.) Haven't validated Tschuprow

    Bias correction and formula's taken from : https://www.researchgate.net/publication/270277061_A_bias-correction_for_Cramer's_V_and_Tschuprow's_T

    Wikipedia for Cramer's V: https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    Wikipedia for Tschuprow' T: https://en.wikipedia.org/wiki/Tschuprow%27s_T
    Parameters:
    -----------
    x : list / ndarray / Pandas Series
        A sequence of categorical measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    bias_correction : Boolean, default = True
    tschuprow : Boolean, default = False
               For choosing Tschuprow as measure
    Returns:
    --------
    float in the range of [0,1]
    """
    corr_coeff = numpy.nan
    try:
        x, y = fill_na(x), fill_na(y)
        crosstab_matrix = pd.crosstab(x, y)
        n_observations = crosstab_matrix.sum().sum()

        yates_correct = True
        if bias_correction:
            if crosstab_matrix.shape == (2, 2):
                yates_correct = False

        chi2, _, _, _ = stats.chi2_contingency(
            crosstab_matrix, correction=yates_correct
        )
        phi2 = chi2 / n_observations

        # r and c are number of categories of x and y
        r, c = crosstab_matrix.shape
        if bias_correction:
            phi2_corrected = max(0, phi2 - ((r - 1) * (c - 1)) / (n_observations - 1))
            r_corrected = r - ((r - 1) ** 2) / (n_observations - 1)
            c_corrected = c - ((c - 1) ** 2) / (n_observations - 1)
            if tschuprow:
                corr_coeff = numpy.sqrt(
                    phi2_corrected / numpy.sqrt((r_corrected - 1) * (c_corrected - 1))
                )
                return corr_coeff
            corr_coeff = numpy.sqrt(
                phi2_corrected / min((r_corrected - 1), (c_corrected - 1))
            )
            return corr_coeff
        if tschuprow:
            corr_coeff = numpy.sqrt(phi2 / numpy.sqrt((r - 1) * (c - 1)))
            return corr_coeff
        corr_coeff = numpy.sqrt(phi2 / min((r - 1), (c - 1)))
        return corr_coeff
    except Exception as ex:
        print(ex)
        if tschuprow:
            warnings.warn("Error calculating Tschuprow's T", RuntimeWarning)
        else:
            warnings.warn("Error calculating Cramer's V", RuntimeWarning)
        return corr_coeff

def cat_cont_correlation_ratio(categories, values):
    """
    Correlation Ratio: https://en.wikipedia.org/wiki/Correlation_ratio
    SOURCE:
    1.) https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
    :param categories: Numpy array of categories
    :param values: Numpy array of values
    :return: correlation
    """
    f_cat, _ = pd.factorize(categories)
    cat_num = numpy.max(f_cat) + 1
    y_avg_array = numpy.zeros(cat_num)
    n_array = numpy.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = values[numpy.argwhere(f_cat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = numpy.average(cat_measures)
    y_total_avg = numpy.sum(numpy.multiply(y_avg_array, n_array)) / numpy.sum(n_array)
    numerator = numpy.sum(
        numpy.multiply(
            n_array, numpy.power(numpy.subtract(y_avg_array, y_total_avg), 2)
        )
    )
    denominator = numpy.sum(numpy.power(numpy.subtract(values, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = numpy.sqrt(numerator / denominator)
    return eta


def type_sep(predictor_list, dataframe):
    cont_cont_preds1 = []  # done
    cont_cont_preds2 = []  # done
    cont_cont_pearsons = []  # done
    cont_cont_html = []  # done
    #     cont_cont_preds1_values = [] #done
    #     cont_cont_preds2_values = [] #done

    cat_cont_preds1 = []
    cat_cont_preds2 = []
    cat_cont_html = []
    for reg_predictor, reg_type in predictor_list:
        for sorted_pred, sorted_type in predictor_list[1:]:
            if reg_type == "continuous" and sorted_type == "continuous":
                pearsons_r = stats.pearsonr(dataframe[reg_predictor].values, dataframe[sorted_pred].values)
                cont_cont_preds1.append(reg_predictor)
                cont_cont_preds2.append(sorted_pred)
                cont_cont_pearsons.append(pearsons_r[0])
                fig = px.scatter(dataframe, x=dataframe[reg_predictor].values, y=dataframe[sorted_pred].values,
                                 trendline="ols")
                fig.update_layout(title=f"chart{reg_predictor}_{sorted_pred}",
                                  xaxis_title=f"Variable: {reg_predictor}", yaxis_title=f"Variable:{sorted_pred}")
                html = "C:/Users/thoma\OneDrive/Documents/bda602/midterm/html_links/{0}_{1}_file.html".format(
                    reg_predictor, sorted_pred)
                fig.write_html(html)
                cont_cont_html.append(html)

                #             elif reg_type == "continuous" and sorted_type == "categorical" or reg_type == "categorical" and sorted_type == "continuous":  :
                #                 cat_preds = np.array([sorted_pred])
                #                 cont_preds = np.array([reg_predictor])
                #                 cat_cont_array = np.concatenate((cat_preds,cont_preds))
                #                 cat_array = dataframe[sorted_pred].to_numpy()
                #                 flattened_cat = cat_array.flatten()

                #                 cat_array = dataframe[sorted_pred].values.flatten()
                #                 le = preprocessing.LabelEncoder()
                #                 le.fit([dataframe[sorted_pred].flatten()])
                #                 classes = le.classes_
                #                 transformed_predictors = le.transform(classes)
                fig1 = px.scatter(dataframe, x=dataframe[reg_predictor].values, y=dataframe[sorted_pred].values,
                                  trendline="ols")
                fig1.update_layout(title=f"chart{reg_predictor}_{sorted_pred}",
                                   xaxis_title=f"Variable: {reg_predictor}", yaxis_title=f"Variable:{sorted_pred}")
                html_cat_cont = "C:/Users/thoma\OneDrive/Documents/bda602/midterm/html_links/{0}_{1}_file.html".format(
                    reg_predictor, sorted_pred)
                fig1.write_html(html_cat_cont)
                cat_cont_html.append(html)

            #                 cat_cont_values = np.array([dataframe[reg_predictors],dataframe[sorted_pred]])
    #                 eta = cat_cont_correlation_ratio(cat_cont_categories,cat_cont_values)

    #             else:

    #             x = dataframe[sorted_pred]
    #             le = preprocessing.LabelEncoder()
    #             le.fit([cat_array])
    #             classes = le.classes_
    #             transformed_predictors = le.transform(classes)
    #             transformed_predictors

    cont_cont_df = pd.DataFrame(
        {"Predictor 1": cont_cont_preds1, "Predictor 2": cont_cont_preds2, "Pearsons R": cont_cont_pearsons,
         "HTML_LinregGraph": cont_cont_html})
    cont_cont_html = cont_cont_df.to_html()
    text_file = open("C:/Users/thoma\OneDrive/Documents/bda602/midterm/html_links/test_dataframe.html", "w")
    text_file.write(cont_cont_html)
    text_file.close()

    return cont_cont_df


cont_cont_df = type_sep(predictor_list, dataframe)


data = cont_cont_df["Pearsons R"].values
heatmap = px.imshow(data,labels=dict(x=cont_cont_df["Predictor 1"], y=cont_cont_df["Predictor 2"]))
heatmap.show()


def main():
    
    return

if __name__ == "__main__":
    sys.exit(main())

