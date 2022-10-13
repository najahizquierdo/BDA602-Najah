import sys

import pandas as pd
import statsmodels.api
from plotly import express as px
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# check if response is cont or bool
def response_bool_or_cont(col):
    response = col.target_names.tolist()
    if response is bool or str:
        return True
    elif len(response) == 2:
        return True
    else:
        return False


def main():
    # import dataset
    data = load_breast_cancer()
    df = pd.DataFrame(data.data)
    columns = data.feature_names
    X = data.data
    y_targ = data.target
    # determine if response is cont or bool
    if response_bool_or_cont(data) is True:
        response = "boolean"
        print("This is a boolean.")
    else:
        response = "continuous"
        print("This is continuous.")

    # linear regression
    if response == "continuous":
        for a, col in enumerate(X.T):
            print(f"Running logistic regression for: {response} ")
            col = X[:, a]
            feature_name = data.feature_names[a]
            predictor = statsmodels.api.add_constant(col)
            linear_regression_model = statsmodels.api.OLS(y_targ, predictor)
            linear_regression_fitted = linear_regression_model.fit()
            print(f"Variable: {feature_name}")
            print(linear_regression_fitted.summary())
            t_value = round(linear_regression_fitted.tvalues[1], 6)
            p_value = "{:.6e}".format(linear_regression_fitted.pvalues[1])
            print(f"T-value: {t_value}")
            fig = px.scatter(x=col, y=y_targ, trendline="ols")
            fig.update_layout(
                title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",
                xaxis_title=f"Variable: {feature_name}",
                yaxis_title="y",
            )
            # fig.show()    # Categorical
    elif response == "boolean":
        # logistic regression
        for a, col in enumerate(X.T):
            print(f"Running logistic regression for: {response} ")
            col = X[:, a]
            feature_name = data.feature_names[a]
            predictor = statsmodels.api.add_constant(col)
            logi_regression_model = statsmodels.api.Logit(
                y_targ, predictor, missing="drop"
            )
            logi_regression_fitted = logi_regression_model.fit()
            print(f"Variable: {feature_name}")
            print(logi_regression_fitted.summary())

            # stats
            t_value = round(logi_regression_fitted.tvalues[1], 6)
            p_value = "{:.6e}".format(logi_regression_fitted.pvalues[1])
            print(f"T-value: {t_value}")
            # plot
            fig = px.scatter(x=col, y=y_targ, trendline="ols")
            fig.update_layout(
                title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",
                xaxis_title=f"Variable: {feature_name}",
                yaxis_title=f"{response}",
            )
            fig.show()


if __name__ == "__main__":
    sys.exit(main())
