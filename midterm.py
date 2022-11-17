import numpy as np
import pandas as pd
import statsmodels.api
from plotly import express as px
from plotly import graph_objects as go
from sklearn.ensemble import RandomForestClassifier


def correlation_metrics(x):
    pearsoncorr = x.corr(method="pearson")
    fig = px.imshow(pearsoncorr)
    fig.show()


def random_forest_var_imp(df, res, cols):
    forest = RandomForestClassifier(random_state=0)
    forest.fit(df[cols], res)
    imp = forest.feature_importances_
    for_imp = pd.Series(imp, index=cols)
    forest_imp_sort = for_imp.sort_values(ascending=True)
    return forest_imp_sort


# def linear_regression(df, response):
#     predictor = statsmodels.api.add_constant(df['strikeout_to_walk_ratio'])
#     linear_regression_model = statsmodels.api.OLS(response, predictor)
#     linear_regression_model_fitted = linear_regression_model.fit()
#     t_value = round(linear_regression_model_fitted.tvalues[1], 6)
#     p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])
#     filename = f"plots/{predictor}_{response}_linear_regression.html"
#     # Plot the figure
#     fig = px.scatter(
#         data_frame=df,
#         x=predictor,
#         y=response,
#         trendline="ols",
#         title=f"(t-value={t_value}) (p-value={p_value})",
#     )


def df_processing(df, response):
    # -- DETERMINE PREDICTORS/ SPLIT --
    df.dropna()
    col = df.columns
    numerical_cols = df._get_numeric_data().columns
    numerical_cols = numerical_cols.drop(["game_id", "binary_winner"])
    x = list(set(col) - set(numerical_cols))
    y = list(df.select_dtypes(bool))
    cat_comparisons = x + y
    # dont think i need this part but i'll leave it in to make me feel like i've done more lol
    cont_predictors = list()
    cat_predictors = list()
    for i in df:
        if i in cat_comparisons:
            print(f"{i} is categorical")
            cat_predictors.append(i)

        else:
            print(f"{i} is continuous")
            p = 1
            cont_predictors.append(i)
    CONT_COLUMNS = list()
    CAT_COLUMNS = list()
    for colName, i in df.iteritems():
        if colName in cat_predictors:
            CAT_COLUMNS.append(i)
        else:
            CONT_COLUMNS.append(i)
    cont_resp = pd.DataFrame(CONT_COLUMNS)
    print(cont_resp)
    cat_resp = pd.DataFrame(CAT_COLUMNS)
    print(cat_resp)
    cont_resp_T = cont_resp.T
    correlation_metrics(cont_resp_T)

    for a in numerical_cols:
        feature_data = df[a]
        predictor = statsmodels.api.add_constant(feature_data)
        linear_regression_model = statsmodels.api.OLS(response, predictor)
        linear_regression_model_fitted = linear_regression_model.fit()
        print(linear_regression_model_fitted.summary())
        t_value = round(linear_regression_model_fitted.tvalues[1], 6)
        p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])
        fig1 = px.histogram(
            df,
            x=feature_data,
            color=response,
        )
        fig1.update_layout(
            title="histogram",
            xaxis_title="Response=" + "Home Team Win",
            yaxis_title="Predictor=" + a,
        )
        fig1.show()
        fig_2 = go.Figure(
            data=go.Violin(
                x=response,
                y=feature_data,
                fillcolor="pink",
                opacity=0.7,
            )
        )
        fig_2.update_layout(
            yaxis_zeroline=False,
            title="histogram",
            xaxis_title="Response=" + "Home Team Win",
            yaxis_title="Predictor=" + a,
        )
        fig_2.show()
