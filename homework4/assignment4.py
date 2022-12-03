import sys

import numpy as np
import pandas as pd
import statsmodels.api
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


# check if response is cont or bool
def response_bool_or_cont(col):
    targ = []
    for i in col:
        targ.append(i)
    targ_df = pd.DataFrame(targ)
    if col is bool and str or np.unique(targ_df).size / targ_df.size < 0.05:
        return True
    else:
        return False


# this is super incorrect ask for help @ office hours
# def diff_mean_of_response(predictor, response):
#     pred_min = predictor.max()
#     pred_max = predictor.min()
#     pop_mean = response.mean()
#     while i <= count(13):
#         bin_size = (pred_min - pred_max)
#         sqr_bin = math.pow(bin_size,2)
#     fig = px.histogram(df, x=sqr_bin, y=pop_mean)


def main():
    # import dataset
    data = load_diabetes()
    df = pd.DataFrame(data=data["data"], columns=data["feature_names"])
    X = data.data
    X_df = pd.DataFrame(X)
    X_df_arr = np.array(X_df)
    t = data["target"]
    target = pd.DataFrame(t)
    # determine if response is cont or bool
    if response_bool_or_cont(t) is True:
        response = 1  # categorical
        print("Response is boolean.")
    else:
        response = 0  # continuous
        print("Response is continuous.")
    # determine if predictor is cat or continuous
    predictors = []
    for a in df.columns:
        if response_bool_or_cont(a) is True:
            predictor_type = 1  # categorical
            predictors.append(a)
            print(f"{a} is categorical")
        else:
            predictor_type = 0
            predictors.append(a)
            cont_predictors = pd.DataFrame(predictors)
            cont_X = np.array(cont_predictors)
            print(f"{a} is continuous")

    # plots
    # CON RESPONSE / CON PREDICT
    # done
    if response == 0 and predictor_type == 0:
        fig = px.scatter(x=X_df_arr[:, 1], y=t, trendline="ols")
        fig.update_layout(
            title="Continuous Response by Continuous Predictor",
            xaxis_title="Predictor",
            yaxis_title="Response",
        )
        fig.show()

    # con response with cat predictor
    if response == 0 and predictor_type == 1:
        # Group data together
        hist_data = X_df_arr[:, 1]
        group_labels = df.columns
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
                    x=np.repeat(curr_group),
                    y=curr_hist,
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
    # cat response with cat predictor
    if response == 1 and predictor_type == 1:
        conf_matrix = confusion_matrix(X_df_arr, df.columns)
        fig_no_relationship = go.Figure(
            data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
        )
        fig_no_relationship.update_layout(
            title="Categorical Predictor by Categorical Resonse (with relationship)",
            xaxis_title="Response",
            yaxis_title="Predictor",
        )
        fig_no_relationship.show()

    # cat response with con predictor
    if response == 1 and predictor_type == 0:
        hist_data = X_df_arr[:, 1]
        group_labels = df.columns
        fig_1 = ff.create_distplot(hist_data, group_labels, bin_size=0.2)
        fig_1.update_layout(
            title="Continuous Predictor by Categorical Response",
            xaxis_title="Predictor",
            yaxis_title="Distribution",
        )
        fig_1.show()
        fig_2 = go.Figure()
        for curr_hist, curr_group in zip(hist_data, group_labels):
            fig_2.add_trace(
                go.Violin(
                    x=np.repeat(curr_group),
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

    # linear regression
    if predictor_type == 0:
        for a, col in enumerate(X_df_arr.T):
            print(f"Running logistic regression for: {response} ")
            col = X_df_arr[:, a]
            feature_name = data.feature_names[a]
            predictor = statsmodels.api.add_constant(col)
            linear_regression_model = statsmodels.api.OLS(target, predictor)
            linear_regression_fitted = linear_regression_model.fit()
            print(f"Variable: {feature_name}")
            print(linear_regression_fitted.summary())
            t_value = round(linear_regression_fitted.tvalues[1], 6)
            p_value = "{:.6e}".format(linear_regression_fitted.pvalues[1])
            print(f"T-value: {t_value}")
            fig = px.scatter(x=col, y=t, trendline="ols")
            fig.update_layout(
                title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",
                xaxis_title=f"Variable: {feature_name}",
                yaxis_title="y",
            )
            fig.show()  # Categorical
    elif predictor_type == 1:
        # logistic regression
        for a, col in enumerate(X_df_arr.T):
            print(f"Running logistic regression for: {response} ")
            col = X_df_arr[:, a]
            feature_name = data.feature_names[a]
            predictor = statsmodels.api.add_constant(col)
            logi_regression_model = statsmodels.api.Logit(
                target, predictor, missing="drop"
            )
            logi_regression_fitted = logi_regression_model.fit()
            print(f"Variable: {feature_name}")
            print(logi_regression_fitted.summary())

            # stats
            t_value = round(logi_regression_fitted.tvalues[1], 6)
            p_value = "{:.6e}".format(logi_regression_fitted.pvalues[1])
            print(f"T-value: {t_value}")
            # plot
            fig = px.scatter(x=col, y=t, trendline="ols")
            fig.update_layout(
                title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",
                xaxis_title=f"Variable: {feature_name}",
                yaxis_title=f"{response}",
            )
            fig.show()
        # random forest
        rand_tree = RandomForestClassifier(n_estimators=10)
        rand_tree.fit(cont_X, t)
        print(rand_tree.predict(cont_X))


if __name__ == "__main__":
    sys.exit(main())
