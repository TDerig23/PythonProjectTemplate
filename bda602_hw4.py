import pandas as pd
import sys
import numpy
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from sklearn.metrics import confusion_matrix

titanic_df = pd.read_csv(
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")


## this is where I will separate the column of an inputted dataset into categorical and continious.
def columnsep():


cat_df = []
cont_df = []
for col in titanic_df:
    if titanic_df[col].dtypes == "object" or titanic_df[col].dtypes == "bool":
        extracted_cat = titanic_df[col]
        cat_df = cat_df.append(extracted_cat)
    else:
        extracted_cont = titanic_df[col]
        cont_df = cont_df.append(extracted_cont)


def cont_resp_cat_predictor(dataframe, cat_predictor, labels):
    # Add histogram data

    # Group data together

    group_labels = dataframe[cat_predictor].values

    # Create distribution plot with custom bin_size
    fig_1 = ff.create_distplot(dataframe, group_labels, bin_size=0.2)
    fig_1.update_layout(
        title="Continuous Response by Categorical Predictor",
        xaxis_title="Response",
        yaxis_title="Distribution",
    )
    fig_1.show()
    fig_1.write_html(
        file="../../../plots/lecture_6_cont_response_cat_predictor_dist_plot.html",
        include_plotlyjs="cdn",
    )

    fig_2 = go.Figure()
    for curr_hist, curr_group in zip(dataframe, group_labels):
        fig_2.add_trace(
            go.Violin(
                x=numpy.repeat(curr_group),
                y=dataframe,
                name=curr_group,
                box_visible=True,
                meanline_visible=True,
            )
        )
    fig_2.update_layout(
        title="Continuous Response by Categorical Predictor",
        xaxis_title="Groupings",
        yaxis_title="Response",
    )
    fig_2.show()

    return


def cat_resp_cont_predictor(dataframe, cont_predictor):
    # Group data together
    hist_data = [x1,

                 group_labels = ["Response = 0", "Response = 1"]

    # Create distribution plot with custom bin_size
    fig_1 = ff.create_distplot(hist_data, group_labels, bin_size=0.2)
    fig_1.update_layout(
        title="Continuous Predictor by Categorical Response",
        xaxis_title="Predictor",
        yaxis_title="Distribution",
    )
    fig_1.show()
    fig_1.write_html(
        file="../../../plots/lecture_6_cat_response_cont_predictor_dist_plot.html",
        include_plotlyjs="cdn",
    )

    fig_2 = go.Figure()
    for curr_hist, curr_group in zip(hist_data, group_labels):
        fig_2.add_trace(
            go.Violin(
                x=numpy.repeat(curr_group, n),
                y=curr_hist,
                name=curr_group,
                box_visible=True,
                meanline_visible=True,
            )
        )
    fig_2.update_layout(
        title="Continuous Predictor by Categorical Response",
        xaxis_title="Response",
        yaxis_title="Predictor",
    )
    fig_2.show()
    fig_2.write_html(
        file="../../../plots/lecture_6_cat_response_cont_predictor_violin_plot.html",
        include_plotlyjs="cdn",
    )
    return


def cat_response_cat_predictor(dataframe, cat_predictor):
    conf_matrix = confusion_matrix(x_2, y_2)

    fig_no_relationship = go.Figure(
        data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
    )
    fig_no_relationship.update_layout(
        title="Categorical Predictor by Categorical Response (without relationship)",
        xaxis_title="Response",
        yaxis_title="Predictor",
    )
    fig_no_relationship.show()
    fig_no_relationship.write_html(
        file="../../../plots/lecture_6_cat_response_cat_predictor_heat_map_no_relation.html",
        include_plotlyjs="cdn",
    )

    x = numpy.random.randn(n)
    y = x + numpy.random.randn(n)

    x_2 = [1 if abs(x_) > 1.5 else 0 for x_ in x]
    y_2 = [1 if abs(y_) > 1.5 else 0 for y_ in y]

    conf_matrix = confusion_matrix(x_2, y_2)

    fig_no_relationship = go.Figure(
        data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
    )
    fig_no_relationship.update_layout(
        title="Categorical Predictor by Categorical Response (with relationship)",
        xaxis_title="Response",
        yaxis_title="Predictor",
    )
    fig_no_relationship.show()
    fig_no_relationship.write_html(
        file="../../../plots/lecture_6_cat_response_cat_predictor_heat_map_yes_relation.html",
        include_plotlyjs="cdn",
    )
    return


def cont_response_cont_predictor(dataframe, cont_predictor):
    fig = px.scatter(x=x, y=y, trendline="ols")
    fig.update_layout(
        title="Continuous Response by Continuous Predictor",
        xaxis_title="Predictor",
        yaxis_title="Response",
    )
    fig.show()
    fig.write_html(
        file="../../../plots/lecture_6_cont_response_cont_predictor_scatter_plot.html",
        include_plotlyjs="cdn",
    )

    return


def datasetregression():
    titanic_df = pd.read_csv(
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")
    X = titanic_df.data
    y = titanic_df.target

    for idx, column in enumerate(X.T):
        feature_name = diabetes.feature_names[idx]
        predictor = statsmodels.api.add_constant(column)
        linear_regression_model = statsmodels.api.OLS(y, predictor)
        linear_regression_model_fitted = linear_regression_model.fit()
        print(f"Variable: {feature_name}")
        print(linear_regression_model_fitted.summary())

        # Get the stats
        t_value = round(linear_regression_model_fitted.tvalues[1], 6)
        p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])

        # Plot the figure
        fig = px.scatter(x=column, y=y, trendline="ols")
        fig.update_layout(
            title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",
            xaxis_title=f"Variable: {feature_name}",
            yaxis_title="y",
        )
        fig.show()
        fig.write_html(
            file=f"../../plots/lecture_6_var_{idx}.html", include_plotlyjs="cdn"
        )


def main():
    cont_resp_cat_predictor()
    cat_resp_cont_predictor()
    cat_response_cat_predictor()
    cont_response_cont_predictor()
    datasetregression()
    return


if __name__ == "__main__":
    sys.exit(main())
