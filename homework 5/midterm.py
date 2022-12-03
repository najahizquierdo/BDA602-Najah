from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api
from plotly import express as px
from plotly import graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def correlation_metrics(x):
    pearsoncorr = x.corr(method="pearson")
    fig = px.imshow(pearsoncorr, color_continuous_scale=px.colors.sequential.RdBu)
    fig.show()


def feature_importance(X, y, cols):
    forest = RandomForestClassifier(n_estimators=150, random_state=0)
    forest.fit(X, y)
    imp = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    idx = np.argsort(imp)[::-1]
    labels = []
    for f in range(X.shape[1]):
        labels.append(cols[f])
    plt.title("Feature Importance:")
    plt.bar(range(X.shape[1]), imp[idx], color="y", yerr=std[idx])
    plt.xticks(range(X.shape[1]), labels, rotation="vertical")
    plt.show()


def brute_force(X, y):
    for pred1, pred2 in combinations(X.columns, 2):
        data = pd.DataFrame()
        pred_list = []
        pred1_df = pd.cut(X[pred1], bins=10)
        pred2_df = pd.cut(X[pred2], bins=10)
        data["response"] = y
        data["Pred1_data"] = pred1_df
        data["Pred2_data"] = pred2_df
        for a, val in data.groupby(["Pred1_data", "Pred2_data"]):
            mean_calc = np.array(val["response"]).mean()
            pred_list.append([a[0], a[1], mean_calc])
            df_list = pd.DataFrame(
                pred_list, columns=["Pred1_data", "Pred2_data", "mean_calc"]
            )
            final_df = df_list.pivot(
                index="Pred1_data", columns="Pred2_data", values="mean_calc"
            )
        b_fig = go.Figure()
        b_fig.add_trace(
            go.Heatmap(
                x=final_df.columns.astype("str"),
                y=final_df.index.astype("str"),
                z=np.array(final_df),
                colorscale="RdBu",
            )
        )
        b_fig.update_layout(title=f"{pred1} vs {pred2}")
        b_fig.show()


def mean_of_response(X, y):
    for pred1 in X.columns:
        data = pd.DataFrame()
        list_1 = []
        pred1_df = pd.cut(X[pred1], bins=8)
        data["Pred1"] = pred1_df
        data["response"] = y
        for a, val in data.groupby(["Pred1"]):
            calculation = np.array(val["response"])
            mor = np.mean(calculation)
            list_1.append([mor, a, len(val)])
            list_2 = pd.DataFrame(list_1, columns=["d_mean", "Pred1", "pop_mean"])
            list_2["bin"] = list_2["Pred1"].apply(lambda x: x.mid)
            list_2["pop_mean"] = np.mean(y)
        m_fig = go.Figure(
            layout=go.Layout(
                title=f"Binned Mean of Response for {pred1}",
                yaxis2=dict(overlaying="y"),
            )
        )
        m_fig.add_trace(go.Bar(x=list_2["bin"], y=list_2["pop_mean"], yaxis="y"))
        m_fig.add_trace(
            go.Scatter(
                x=list_2["bin"],
                y=list_2["d_mean"],
                yaxis="y",
                mode="lines",
                line=go.scatter.Line(color="pink"),
            )
        )
        m_fig.add_trace(
            go.Scatter(
                x=list_2["bin"],
                y=list_2["pop_mean"],
                yaxis="y2",
                mode="lines",
                line=go.scatter.Line(color="green"),
            )
        )
        m_fig.show()


def linear_regression(df, numerical_cols, response):
    for a in numerical_cols:
        feature_data = df[a]
        predictor = statsmodels.api.add_constant(feature_data)
        linear_regression_model_fitted = statsmodels.api.OLS(response, predictor).fit()
        # print(linear_regression_model_fitted.summary())
        t_value = round(linear_regression_model_fitted.tvalues[1], 6)
        p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])
        lr_fig = px.scatter(x=feature_data, y=response, trendline="ols")
        lr_fig.update_layout(
            title=f"(Variable: {a} and {df.columns[1]}, t-value={t_value}) (p-value={p_value})",
            xaxis_title=f"Variable: {a}",
            yaxis_title=str(df.columns[1]),
        )
        # lr_fig.show()


def cont_plots(df, numerical_cols, response):
    y = df[response]
    for a in numerical_cols:
        # distribution plot
        plt.figure(figsize=(12, 8))
        sns.distplot(df[a], bins=10)
        plt.title("Distribution by" + f"{a}")
        plt.xlabel(f"{a}")
        # plt.show();
        # histogram
        hist_fig = px.histogram(
            df,
            x=df[a],
            color=y,
        )
        hist_fig.update_layout(
            title="histogram",
            xaxis_title="Response=" + f"{response}",
            yaxis_title="Predictor=" + f"{a}",
        )
        # hist_fig.show()
        # violin plot
        v_plot = go.Figure(
            data=go.Violin(
                x=y,
                y=df[a],
                fillcolor="pink",
                opacity=0.7,
            )
        )
        v_plot.update_layout(
            yaxis_zeroline=False,
            title="Violin",
            xaxis_title="Response=" + f"{response}",
            yaxis_title="Predictor=" + a,
        )
        # v_plot.show()


def df_processing(df, response):
    df = df.dropna()
    # drop game_id and binary for usage
    numerical_cols = df._get_numeric_data().columns.drop("game_id")
    numerical_cols = numerical_cols.drop(response)
    X = df[numerical_cols]
    y = df[response]
    # correlation
    correlation_metrics(df)
    # feature importance
    feature_importance(X, y, numerical_cols)
    # linear regression
    linear_regression(df, numerical_cols, y)
    # plots
    cont_plots(df, numerical_cols, response)
    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0
    )
    # brute force
    brute_force(X_train, y_train)
    # mean of resp
    mean_of_response(X_train, y_train)
