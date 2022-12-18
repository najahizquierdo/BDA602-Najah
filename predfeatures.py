import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy
import base64
from io import BytesIO
from itertools import combinations
from plotly import express as px
from plotly import graph_objects as go
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

def make_clickable(val):
    return '<a target="blank" href="{}">{}</a>'.format(val, val)

def set_html_up():
    text_file2 = open("index.html", "w")
    text_file2.write("<a href='brute_force.html'> <h1>Brute Force Table</h1></a>")
    text_file2.write("<a href='cont_plots.html'> <h1>Plots</h1></a></h1>")
    text_file2.write("<a href='meanofresp.html'> <h1>Mean of Response</h1></a></h1>")
    text_file2.write("<a href='logistic_reg.html'> <h1> Logistic Regression</h1></a></h1>")
    text_file2.write("<a href='corr_heatmap.html'> <h1> Correlation Matrix </h1></a>")
    text_file2.write("<a href='knn.html'> <h1> k-Nearest Neighbor </h1></a>")
    text_file2.write("<a href='featureimp.html'> <h1> Feature Importance </h1></a>")
    text_file2.close()

# def model_performance(model_name, score):


def correlation_metrics(x):
    x.dropna()
    corr = x.corr(method="pearson")
    fig = px.imshow(corr, color_continuous_scale=px.colors.sequential.RdBu)
    fig.write_html(file=f"corr_heatmap.html", include_plotlyjs="cdn")
    #fig.show();
def cont_plots(df, numerical_cols, response):
    table_df = pd.DataFrame(
        columns=[
            "Predictors",
            "Violin Plot",
            "Histogram",
        ]
    )
    table_df.style.format({"Violin Plot": make_clickable})
    table_df.style.format({"Dist Plot": make_clickable})
    y = df[response]
    for a in numerical_cols:
        # #distribution plot
        # plt.figure(figsize=(12, 8))
        # sns.distplot(df[a], bins=10)
        # plt.title(f"Distribution by" + f"{a}")
        # plt.xlabel(f"{a}")
        # # plt.show();
        #histogram
        hist_fig = px.histogram(
            df,
            x=df[a],
            color=y,
        )
        hist_fig.update_layout(
            title=f"Histogram: {a}",
            xaxis_title="Response=" + f"{response}",
            yaxis_title="Predictor=" + f"{a}",
        )
        hist_fig.write_html(
          file=f"hist_{response}_{a}.html", include_plotlyjs="cdn"
        )
        hist_name = "hist_" + str(response) + "_" + str(a) + ".html"
        #hist_fig.show()
        #violin plot
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
        v_plot.write_html(
          file=f"violin_plot_{response}_{a}.html", include_plotlyjs="cdn"
        )
        violin_name = "violin_plot_" + str(response) + "_" + str(a) + ".html"
        new_row = {
            table_df.columns[0]: f"{a}",
            table_df.columns[1]: make_clickable(violin_name),
            table_df.columns[2]: make_clickable(hist_name)
        }
        table_df = table_df.append(new_row, ignore_index=True)
        cat_cont = table_df.to_html(render_links=True, escape=False)
        text_file = open("cont_plots.html", "w")
        text_file.write(
            f"<h1><center>Continuous Plots</h1>"
            f"<h2><center>Correlation Table </h2> "
            f"<center>{cat_cont}</center>"
            f"<center><a href='corr_heatmap.html'>Heatmap</a>"
        )
        text_file.close()
        #v_plot.show()
def brute_force(X, y):
   brute_df = pd.DataFrame(
        columns=[
            "Predictor 1",
            "Predictor 2",
            "Bin Plot",
        ]
    )
   brute_df.style.format({"Bin Plot": make_clickable})
   X = X.dropna()
   y = y.dropna()
   for pred1, pred2 in combinations(X.columns, 2):
        data = pd.DataFrame()
        pred_list = []
        pred1_df = pd.cut(X[pred1], bins = 10)
        pred2_df = pd.cut(X[pred2], bins =10)
        data["response"] = y
        data["Pred1_data"] = pred1_df
        data["Pred2_data"] = pred2_df
        for a, val in data.groupby(["Pred1_data", "Pred2_data"]):
            mean_calc = np.array(val["response"]).mean()
            pred_list.append([a[0], a[1], mean_calc])
            df_list = pd.DataFrame(pred_list, columns=["Pred1_data", "Pred2_data", "mean_calc"])
            final_df = df_list.pivot(
                index="Pred1_data",
                columns="Pred2_data",
                values="mean_calc"
            )
        b_fig = go.Figure()
        b_fig.add_trace(
            go.Heatmap(
                 x=final_df.columns.astype("str"),
                 y=final_df.index.astype("str"),
                 z=np.array(final_df),
                colorscale='RdBu'
            )
        )
        b_fig.update_layout(
            title=f'{pred1} vs {pred2}'
         )
        b_fig.write_html(
          file=f"brute_force_{pred1}_{pred2}.html", include_plotlyjs="cdn"
        )
        brute_force_plot = "brute_force_" + str(pred1) + "_" + str(pred2) + ".html"
        new_row = {
            brute_df.columns[0]: f"{pred1}",
            brute_df.columns[1]: f"{pred2}",
            brute_df.columns[2]: make_clickable(brute_force_plot)
        }
        brute_df = brute_df.append(new_row, ignore_index=True)
        brute_force_html = brute_df.to_html(render_links=True, escape=False)
        text_file = open("brute_force.html", "w")
        text_file.write(
            f"<h1><center>Brute Force</h1>"
            f"<center>{brute_force_html}</center>"
        )
        text_file.close()
def mean_of_response(X, y):
    mor_df = pd.DataFrame(
        columns=[
            "Predictor",
            "Mean Squared Difference",
            "Mean of Response Plot",
        ]
    )
    mor_df.style.format({"Mean of Response Plot": make_clickable})
    for pred1 in X.columns:
        # plt.hist(X[pred1], bins=10)
        # plt.title(f"Binned Mean of Resp {pred1}")
        # plt.show()
        pred1_df = pd.qcut(X[pred1], q=10, duplicates = 'drop')
        data = pd.DataFrame()
        list_1 = []
        data["Pred1"] = pred1_df
        data["response"] = y
        for a, val in data.groupby(["Pred1"]):
            calculation = np.array(val["response"])
            mor = np.mean(calculation)
            list_1.append([mor, a, len(val)])
            list_2 = pd.DataFrame(list_1, columns=["d_mean", "Pred1", "pop_mean"])
            list_2["bin"] = list_2["Pred1"].apply(lambda x: x.mid)
            list_2["pop_mean"] = np.mean(y)
            mse = mean_squared_error(list_2["bin"], list_2["pop_mean"])
            mse = mse.round(decimals=4)
        m_fig = go.Figure(
            layout=go.Layout(
                title=f"Binned Mean of Response for {pred1}",
                yaxis2=dict(overlaying="y"),
            )
        )
        m_fig.add_trace(
            go.Bar(
                x=list_2["bin"],
                y=list_2["d_mean"],
                yaxis="y2"
            )
        )
        m_fig.add_trace(
            go.Scatter(
                x=list_2["bin"],
                y=list_2["pop_mean"],
                yaxis="y1",
                mode="lines",
                line=go.scatter.Line(color="pink"),
            )
        )
        m_fig.add_trace(
            go.Scatter(
                x=list_2["bin"],
                y=list_2["d_mean"],
                yaxis="y2",
                mode="lines",
                line=go.scatter.Line(color="green"),
            )
        )
        m_fig.write_html(
          file=f"mean_of_resp_{pred1}.html", include_plotlyjs="cdn"
        )
        mr_plot = "mean_of_resp_" + str(pred1) + ".html"
        new_row = {
            mor_df.columns[0]: f"{pred1}",
            mor_df.columns[1]: f"{mse}",
            mor_df.columns[2]: make_clickable(mr_plot),
        }
        mor_df = mor_df.append(new_row, ignore_index=True)
        mr_html = mor_df.to_html(render_links=True, escape=False)
        text_file = open("meanofresp.html", "w")
        text_file.write(
            f"<h1><center>Mean of Response</h1>"
            f"<center>{mr_html}</center>"
        )
        text_file.close()
        # m_fig.show()
#MODELS
def logreg(X_test, y_test, X, y, numerical_cols):
    log_df = pd.DataFrame(
        columns=[
            "Predictor",
            "t-value",
            "p-value",
            "Logistic Regression Plot",
        ]
    )
    log_df.style.format({"Logistic Regression Plot": make_clickable})
    for a in numerical_cols:
        pred = sm.add_constant(X[a])
        logistic_regression_model = sm.Logit(y, pred)
        logistic_regression_model_fitted = logistic_regression_model.fit()
        t_value = round(logistic_regression_model_fitted.tvalues[1], 4)
        p_value = round(logistic_regression_model_fitted.pvalues[1], 4)
        print(logistic_regression_model_fitted.summary())
        print(f"t-value: {t_value} , p-value: {p_value}" )
        fig = px.scatter(x=X[a], y=y, trendline="ols")
        fig.update_layout(
            title=f"Variable: {a}: (t-value={t_value}) (p-value={p_value})",
            xaxis_title=f"Variable: {a}",
            yaxis_title="y",
        )
        # fig.show()
        fig.write_html(
          file=f"log_reg_{a}.html", include_plotlyjs="cdn"
        )
        log_plot = "log_reg_" + str(a) + ".html"
        new_row = {
            log_df.columns[0]: f"{a}",
            log_df.columns[1]: f"{t_value}",
            log_df.columns[2]: f"{p_value}",
            log_df.columns[3]: make_clickable(log_plot)
        }
        log_df = log_df.append(new_row, ignore_index=True)
    log_html = log_df.to_html(render_links=True, escape=False)
    logreg = LogisticRegression()
    logreg.fit(X_test, y_test)
    y_pred = logreg.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    t_file = BytesIO()
    plt.savefig(t_file, format='png')
    encoded = base64.b64encode(t_file.getvalue()).decode('utf-8')
    html = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
    text_file = open("logistic_reg.html", "w")
    text_file.write(
        f"<h1><center> Logistic Regression </h1>"
        f"<center>{log_html}</center>"
        f"<center> <h2>Model Accuracy:<h2> <br><center> {round(logreg.score(X_test, y_test), 4)}"
        f"<center>{html}</center>"
    )
    text_file.close()
def feature_importance(X, y, cols):
    forest = RandomForestClassifier(n_estimators=150, random_state=0)
    forest.fit(X, y)
    i = forest.feature_importances_
    s = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    index = np.argsort(i)[::-1]
    lbl = []
    for f in range(X.shape[1]):
        lbl.append(cols[f])
    plt.title("Feature Importance:")
    plt.bar(range(X.shape[1]), i[index],
            color= '#FB575D', yerr=s[index])
    plt.xticks(range(X.shape[1]), lbl, rotation='vertical')
    t_file = BytesIO()
    plt.savefig(t_file, format='png')
    encoded = base64.b64encode(t_file.getvalue()).decode('utf-8')
    html = 'feature_importance' + '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
    with open('featureimp.html', 'w') as f:
        f.write(html)
    #plt.show();
def tuningRandomizedSearchCV(df, parameters, X, y):
    rand = RandomizedSearchCV(df, parameters, cv=10, scoring='accuracy', n_iter=10, random_state=5)
    rand.fit(X, y)
    rand.cv_results_
    best_scores = []
    for _ in range(20):
        rand = RandomizedSearchCV(df, parameters, cv=10, scoring='accuracy', n_iter=10)
        rand.fit(X, y)
        best_scores.append(round(rand.best_score_, 3))
    print(best_scores)
def knn(X_test, X_train, y_train, y_test, X, y):
    knn = KNeighborsClassifier(n_neighbors=5)
    k_range = list(range(1, 31))
    weight_options = ['uniform', 'distance']
    param_dist = dict(n_neighbors=k_range, weights=weight_options)
    tuningRandomizedSearchCV(knn, param_dist, X, y)
    knn = KNeighborsClassifier(n_neighbors=27, weights='uniform')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    t_file = BytesIO()
    plt.savefig(t_file, format='png')
    encoded = base64.b64encode(t_file.getvalue()).decode('utf-8')
    html = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
    with open('knn.html', 'w') as f:
        f.write(html)
        f.write(f"<br> Model Accuracy: {round(knn.score(X_test, y_test), 5)}")


