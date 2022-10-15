from io import StringIO

import pandas as pd
import sys
import numpy
import statsmodels as statsmodels
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import random


def column_sep(dataframe):
    cat_df = pd.DataFrame()
    cont_df = pd.DataFrame()
    for col in dataframe:
        if dataframe[col].dtypes == "bool" or dataframe[col].dtypes == "object" or len(pd.unique(dataframe[col])) == 2:
            extracted_cat = dataframe[col]
            cat_df = cat_df.append(extracted_cat)
        else:
            extracted_cont = dataframe[col]
            cont_df = cont_df.append(extracted_cont)

    return cat_df, cont_df


def get_column_names(dataframe):
    cat_df, cont_df = column_sep(dataframe)

    return cat_df.info, cont_df.info


def cont_resp_cat_predictor(dataframe):
    cat_df, cont_df = column_sep(dataframe)
    n = random.randint(0, 8)
    # Add histogram data
    hist_data = cat_df.iloc[:, n]
    group_labels = cont_df.iloc[:, n].values
    # Group data together

    # Create distribution plot with custom bin_size
    fig_1 = ff.create_distplot(hist_data, group_labels, bin_size=0.2)
    fig_1.update_layout(
        title="Continuous Response by Categorical Predictor",
        xaxis_title="Response",
        yaxis_title="Distribution",
    )
    fig_1.show()

    fig_2 = go.Figure()
    for curr_hist, curr_group in zip(hist_data, group_labels):
        fig_2.add_trace(
            go.Violin(
                x=numpy.repeat(curr_group, n),
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

    return fig_1, fig_2


def cat_resp_cont_predictor(dataframe):
    cat_df, cont_df = column_sep(dataframe)
    n = random.randint(0, 8)
    # Add histogram data
    hist_data = cont_df.iloc[:, n]
    group_labels = cat_df.iloc[:, n].values
    # Group data together

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


def cat_response_cat_predictor(dataframe):
    cat_df, cont_df = column_sep(dataframe)
    n = random.randint(0, 8)
    # Add histogram data
    x = cat_df.iloc[:, n].values
    y = cont_df.iloc[:, n].values
    # Group data together

    conf_matrix = confusion_matrix(x, y)

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

    conf_matrix = confusion_matrix(x,y)

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


def cont_response_cont_predictor(dataframe):
    cat_df, cont_df = column_sep(dataframe)
    n = random.randint(0, 8)
    # Add histogram data
    x = cat_df.iloc[:, n]
    y = cont_df.iloc[:, n].values
    # Group data together

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


def datasetregression(dataframe):
    cat_df, cont_df = column_sep(dataframe)
    n = random.randint(0, 8)
    # Add histogram data
    hist_data = cat_df.iloc[:, n]
    group_labels = cont_df.iloc
    # Group data together
    X = dataframe.data
    y = dataframe.target

    for idx, column in enumerate(X.T):
        feature_name = dataframe.feature_names[idx]
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


def plot_decision_tree(decision_tree, feature_names, class_names, file_out):
    with StringIO() as dot_data:
        export_graphviz(
            decision_tree,
            feature_names=feature_names,
            class_names=class_names,
            out_file=dot_data,
            filled=True,
        )
        graph = pydot.graph_from_dot_data(dot_data.getvalue())


def decision_tree_setup(dataframe):
    # Increase pandas print viewport (so we see more on the screen)
    pd.set_option("display.max_rows", 60)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1_000)

    cat_df, cont_df = column_sep(dataframe)
    n = random.randint(0, 8)
    # Add histogram data

    column_names = dataframe.columns

    # Drop rows with missing values
    dataframe = dataframe.dropna()

    print("Original Dataset")

    # Continuous Features

    X = dataframe[cont_df].values

    # Response
    y = dataframe.iloc[:, n].values

    # Decision Tree Classifier
    max_tree_depth = 7
    tree_random_state = 0  # Always set a seed
    decision_tree = DecisionTreeClassifier(
        max_depth=max_tree_depth, random_state=tree_random_state
    )
    decision_tree.fit(X, y)

    # Plot the decision tree
    plot_decision_tree(
        decision_tree=decision_tree,
        feature_names=x,
        class_names="classification",
        file_out="../../plots/lecture_6_iris_tree_full",
    )

    # Find an optimal tree via cross-validation
    parameters = {
        "max_depth": range(1, max_tree_depth),
        "criterion": ["gini", "entropy"],
    }
    decision_tree_grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=tree_random_state), parameters, n_jobs=4
    )
    decision_tree_grid_search.fit(X=X, y=y)

    cv_results = DataFrame(decision_tree_grid_search.cv_results_["params"])
    cv_results["score"] = decision_tree_grid_search.cv_results_["mean_test_score"]
    print_heading("Cross validation results")
    print(cv_results)
    print_heading("Cross validation results - HTML table")
    print(cv_results.to_html())

    # Plot these cross_val results
    gini_results = cv_results.loc[cv_results["criterion"] == "gini"]
    entropy_results = cv_results.loc[cv_results["criterion"] == "entropy"]
    data = [
        go.Scatter(
            x=gini_results["max_depth"].values,
            y=gini_results["score"].values,
            name="gini",
            mode="lines",
        ),
        go.Scatter(
            x=entropy_results["max_depth"].values,
            y=entropy_results["score"].values,
            name="entropy",
            mode="lines",
        ),
    ]

    layout = go.Layout(
        title="Fisher's Iris Cross Validation",
        xaxis_title="Tree Depth",
        yaxis_title="Score",
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show()
    fig.write_html(
        file="../../plots/lecture_6_iris_cross_val.html",
        include_plotlyjs="cdn",
    )

    # Get the "best" model
    best_tree_model = decision_tree_grid_search.best_estimator_

    # Plot this "best" decision tree
    plot_decision_tree(
        decision_tree=best_tree_model,
        feature_names=continuous_features,
        class_names="classification",
        file_out="../../plots/lecture_6_iris_tree_cross_val",
    )
    return


def main():
    cont_resp_cat_predictor()
    cat_resp_cont_predictor()
    cat_response_cat_predictor()
    cont_response_cont_predictor()
    datasetregression()
    return


if __name__ == "__main__":
    sys.exit(main())
