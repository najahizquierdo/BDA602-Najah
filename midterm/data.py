# import sys
# import warnings
#
# import numpy as np
# import pandas as pd
#
# warnings.simplefilter(action="ignore", category=FutureWarning)
# import random
#
# import plotly.express as px
# import statsmodels.api
# from plotly import graph_objects as go
# from sklearn.ensemble import RandomForestClassifier
#
#
# def make_clickable(val):
#     return '<a target="blank" href="{}">{}</a>'.format(val, val)
#
#
# def set_html_up():
#     text_file2 = open("index.html", "w")
#     text_file2.write("<a href='cont_cont.html'> <h1>Continous/Continous</h1></a></h1>")
#     text_file2.write("<a href='cont_cat.html'> <h1>Categorical/Continous</h1></a></h1>")
#     text_file2.write(
#         "<a href='cat_cat.html'> <h1>Categorical/Categorical</h1></a></h1>"
#     )
#     text_file2.close()
#
#
# def correlation_metrics(x):
#     pearsoncorr = x.corr(method="pearson")
#     fig = px.imshow(pearsoncorr)
#     fig.write_html(file=f"corr_heatmap_{x.name}.html", include_plotlyjs="cdn")
#     fig.show()
#     return print(pearsoncorr)
#
#
# def cat_cont_correlation_metrics(categorical, continuous):
#     biserial_corr = (
#         pd.concat([categorical, continuous], axis=1, keys=["categorical", "continuous"])
#         .corr()
#         .loc["categorical", "continuous"]
#     )
#     fig2 = px.imshow(biserial_corr)
#     fig2.write_html(file="corr_heatmap_cat_cont.html", include_plotlyjs="cdn")
#     fig2.show()
#     return print(biserial_corr)
#
#
# def cat_cont_table(df, cat, cont, table_df, table_df2):
#     i = 0
#     for a, col in enumerate(df.T):
#         while i <= 15:
#             col = random.choice(cat.columns)
#             feature_name = random.choice(cont.columns)
#             df = df.dropna()
#             y = cat[col]
#             x = cont[feature_name]
#             # violin plot + dist plot
#             fig1 = px.histogram(
#                 df,
#                 x=y,
#                 color=x,
#                 hover_data=df.columns,
#             )
#             fig1.update_layout(
#                 title=f"{col} and {feature_name}",
#                 xaxis_title="Response=" + col,
#                 yaxis_title="Predictor=" + feature_name,
#             )
#             fig1.write_html(
#                 file=f"dist_plot_{col}_{feature_name}.html", include_plotlyjs="cdn"
#             )
#             dist_name = "dist_plot_" + str(col) + "_" + str(feature_name) + ".html"
#             fig1.show()
#             # -- VIOLIN --
#             fig_2 = go.Figure(
#                 data=go.Violin(
#                     x=x,
#                     y=y,
#                     fillcolor="pink",
#                     opacity=0.7,
#                     x0=feature_name,
#                 )
#             )
#             fig_2.update_layout(
#                 yaxis_zeroline=False,
#                 title=f"{col} and {feature_name}",
#                 xaxis_title="Response=" + col,
#                 yaxis_title="Predictor=" + feature_name,
#             )
#             fig_2.write_html(
#                 file=f"violin_plot_{col}_{feature_name}.html", include_plotlyjs="cdn"
#             )
#             violin_name = "violin_plot_" + str(col) + "_" + str(feature_name) + ".html"
#             corr_coeff = np.corrcoef(y, x)[1, 0]
#             fig_2.show()
#             mean_of_resp(df, col, feature_name, table_df2, "cat-cont")
#             new_row = {
#                 table_df.columns[0]: f"{feature_name} and {col}",
#                 table_df.columns[1]: corr_coeff,
#                 table_df.columns[2]: make_clickable(violin_name),
#                 table_df.columns[3]: make_clickable(dist_name),
#             }
#             table_df = table_df.append(new_row, ignore_index=True)
#             cat_cont = table_df.to_html(render_links=True, escape=False)
#             text_file = open("cont_cat.html", "w")
#             text_file.write(
#                 f"<h1><center>Categorical/Continous</h1>"
#                 f"<h2><center>Correlation Table </h2> "
#                 f"<center>{cat_cont}</center>"
#                 f"<center><a href='corr_heatmap_cat_cont.html'>Heatmap</a>"
#             )
#             text_file.close()
#             i += 1
#
#
# def cont_cont_table(cont_resp, table_df, table_df2):
#     i = 0
#     for a, col in enumerate(cont_resp.T):
#         while i <= 6:
#             col = random.choice(cont_resp.columns)
#             feature_name = random.choice(cont_resp.columns)
#             cont_resp = cont_resp.dropna()
#             y = cont_resp[col]
#             x = cont_resp[feature_name]
#             model = statsmodels.api.OLS(y, x)
#             model_fitted = model.fit()
#             print(f"Variable: {col} and {feature_name}")
#             print(model_fitted.summary())
#             t_value = round(model_fitted.tvalues[0], 6)
#             p_value = "{:.6e}".format(model_fitted.pvalues[0])
#             # Plot the figure
#             pearsoncorr = y.corr(x, method="pearson")
#             print(col)
#             name = "linear_regression_" + str(col) + "_" + str(feature_name) + ".html"
#             mean_of_resp(cont_resp, f"{col}", f"{feature_name}", table_df2, "cont-cont")
#             # ONLY PRINTS OUT ACTUAL NAME WHEN I PUT A STOP HERE???
#             fig = px.scatter(x=cont_resp[col], y=x, trendline="ols")
#             fig.update_layout(
#                 title=f"(Variable: {feature_name} and {col}, t-value={t_value}) (p-value={p_value})",
#                 xaxis_title=f"Variable: {feature_name}",
#                 yaxis_title=str(y),
#             )
#             fig.write_html(
#                 file=f"linear_regression_{col}_{feature_name}.html",
#                 include_plotlyjs="cdn",
#             )
#             new_row = {
#                 table_df.columns[0]: f"{feature_name} and {col}",
#                 table_df.columns[1]: pearsoncorr,
#                 table_df.columns[2]: make_clickable(name),
#             }
#             table_df = table_df.append(new_row, ignore_index=True)
#             cont_cont = table_df.to_html(render_links=True, escape=False)
#             text_file4 = open("cont_cont.html", "w")
#             text_file4.write(
#                 f"<h1><center>Continous/Continous</h1>"
#                 f"<h2><center>Correlation Table </h2> "
#                 f"<center>{cont_cont}</center>"
#                 f"<center><a href='corr_heatmap_cont.html'>Heatmap</a>"
#             )
#             text_file4.close()
#             fig.show()
#             i += 1
#
#
# def mean_of_resp(df, pred1, pred2, table_df2, file):
#     pred1name = pred1
#     pred2name = pred2
#     # --GET BIN DATA--
#     bin_data = pd.DataFrame()
#     bin_data = (df.groupby([pred1, pred2])["survived"]).mean()
#     mean_resp = bin_data.mean()
#     name = "heatmap_" + str(pred1name) + "_" + str(pred2name) + ".html"
#     new_row = {
#         table_df2.columns[0]: f"{pred1name}",
#         table_df2.columns[1]: f"{pred2name}",
#         table_df2.columns[2]: mean_resp,
#         table_df2.columns[3]: make_clickable(name),
#         table_df2.columns[4]: make_clickable(name),
#     }
#     table_df2 = table_df2.append(new_row, ignore_index=True)
#     mean_of_res = table_df2.to_html(render_links=True, escape=False)
#     text_file4 = open(f"brute_force_{file}.html", "w")
#     text_file4.write(
#         f"<h1><center>Brute Force</h1>"
#         f"<h2><center>Correlation Table </h2> "
#         f"<center>{mean_of_res}</center>"
#     )
#     text_file4.close()
#
#
# def random_forest_var_imp(df, res, cols):
#     forest = RandomForestClassifier(random_state=0)
#     forest.fit(df[cols], res)
#     imp = forest.feature_importances_
#     for_imp = pd.Series(imp, index=cols)
#     forest_imp_sort = for_imp.sort_values(ascending=True)
#     return forest_imp_sort
#
#
# def main():
#     # --LOADING DATA --
#     df = pd.read_csv(
#         "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
#     )
#     df = df.dropna()
#     set_html_up()
#
#     # -- DETERMINE PREDICTORS/ SPLIT --
#     col = df.columns
#     response = df.survived
#     numerical_cols = df._get_numeric_data().columns
#     x = list(set(col) - set(numerical_cols))
#     y = list(df.select_dtypes(bool))
#     cat_comparisons = x + y
#     cont_predictors = list()
#     cat_predictors = list()
#     for i in df:
#         if i in cat_comparisons:
#             print(f"{i} is categorical")
#             cat_predictors.append(i)
#
#         else:
#             print(f"{i} is continuous")
#             cont_predictors.append(i)
#
#     CONT_COLUMNS = list()
#     CAT_COLUMNS = list()
#     for colName, i in df.iteritems():
#         if colName in cat_predictors:
#             CAT_COLUMNS.append(i)
#         else:
#             CONT_COLUMNS.append(i)
#     cont_resp = pd.DataFrame(CONT_COLUMNS)
#     cat_resp = pd.DataFrame(CAT_COLUMNS)
#     cont_resp = cont_resp.T
#     cat_resp = cat_resp.T
#
#     # --CREATE DF FOR HTML --
#     # CONT/CONT TABLE
#     cont_cont_df = pd.DataFrame(
#         columns=[
#             "Predictors",
#             "Correlation",
#             "Linear Regression",
#         ]
#     )
#     cont_cont_df.style.format({"Linear Regression": make_clickable})
#     # CAT/CONT TABLE
#     cat_cont_df = pd.DataFrame(
#         columns=[
#             "Predictors",
#             "Correlation",
#             "Violin Plot",
#             "Dist Plot",
#         ]
#     )
#     cat_cont_df.style.format({"Violin Plot": make_clickable})
#     cat_cont_df.style.format({"Dist Plot": make_clickable})
#     # CAT/CAT PREDICTORS
#     cat_cat_df = pd.DataFrame(
#         columns=[
#             "Predictors",
#             "Correlation",
#             "Heatmap",
#         ]
#     )
#     cat_cat_df.style.format({"Heatmap": make_clickable})
#     # -- BRUTE FORCE --
#     cont_cont_bf = pd.DataFrame(
#         columns=[
#             "Predictor 1",
#             "Predictor 2",
#             "Diff of Mean of Response",
#             "Bin Plot",
#             "Residuals",
#         ]
#     )
#     cont_cont_bf.style.format({"Bin Plot": make_clickable})
#     cont_cont_bf.style.format({"Residuals": make_clickable})
#
#     cat_cont_bf = pd.DataFrame(
#         columns=[
#             "Predictor 1",
#             "Predictor 2",
#             "Diff of Mean of Response",
#             "Bin Plot",
#             "Residuals",
#         ]
#     )
#     cat_cont_bf.style.format({"Bin Plot": make_clickable})
#     cat_cont_bf.style.format({"Residuals": make_clickable})
#     cat_cat_bf = pd.DataFrame(
#         columns=[
#             "Predictor 1",
#             "Predictor 2",
#             "Diff of Mean of Response",
#             "Bin Plot",
#             "Residuals",
#         ]
#     )
#     cat_cat_bf.style.format({"Bin Plot": make_clickable})
#     cat_cat_bf.style.format({"Residuals": make_clickable})
#     biserial_cat_resp = cat_resp.apply(lambda x: pd.factorize(x)[0])
#     biserial_cat_resp.name = "cat"
#
#     # -- CORRELATION MATRICES--
#     cont_resp.name = "cont"
#     cat_resp.name = "cat"
#
#     # correlation for continuous responses
#     correlation_metrics(cont_resp)
#     # correlation for categorical response
#     correlation_metrics(biserial_cat_resp)
#     # correlation for cont and cat responses
#     cat_cont_correlation_metrics(biserial_cat_resp, cont_resp)
#     # cont-cont predictor pairs
#     random_forest_var_imp(df, response, numerical_cols)
#     cont_cont_table(cont_resp, cont_cont_df, cont_cont_bf)
#     cat_cont_table(df, biserial_cat_resp, cont_resp, cat_cont_df, cat_cont_bf)
#     for a, col in enumerate(biserial_cat_resp.T):
#         col = biserial_cat_resp.columns[a]
#         feature_name = biserial_cat_resp.columns[a + 1]
#         cat_resp = biserial_cat_resp.dropna()
#         y = biserial_cat_resp[col]
#         x = biserial_cat_resp[feature_name]
#         pearsoncorr = np.corrcoef(y, x)
#         fig2 = px.imshow(pearsoncorr)
#         fig2.update_layout(
#             title=f"Correlation Heatmap {col} and {feature_name}",
#             xaxis_title=f"{col}",
#             yaxis_title=f"{feature_name}",
#         )
#         fig2.write_html(
#             file=f"heatmap_{col}_{feature_name}.html", include_plotlyjs="cdn"
#         )
#         name = "heatmap_" + str(col) + "_" + str(feature_name) + ".html"
#         fig2.show()
#         mean_of_resp(
#             biserial_cat_resp, f"{col}", f"{feature_name}", cat_cat_bf, "cat-cat"
#         )
#         pearsoncorr_table = np.corrcoef(y, x)[1, 0]
#         new_row = {
#             cat_cat_df.columns[0]: f"{feature_name} and {col}",
#             cat_cat_df.columns[1]: pearsoncorr_table,
#             cat_cat_df.columns[2]: make_clickable(name),
#         }
#         cat_cat_df = cat_cat_df.append(new_row, ignore_index=True)
#         print(cat_cat_df)
#         cat_cat = cat_cat_df.to_html(render_links=True, escape=False)
#         text_file3 = open("cat_cat.html", "w")
#         text_file3.write(
#             f"<h1><center>Categorical/Categorical</h1>"
#             f"<h2><center>Correlation Table </h2> "
#             f"<center>{cat_cat}</center>"
#             f"<center><a href='corr_heatmap_cat.html'>Heatmap</a>"
#         )
#         text_file3.close()
#
#
# if __name__ == "__main__":
#     sys.exit(main())
