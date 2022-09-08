import numpy as np
import pandas
import plotly.express as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris_df = pandas.read_csv("iris.data", header=None)
iris_df.columns = [
    "Sepal Length(cm)",
    "Sepal Width(cm)",
    "Petal Length(cm)",
    "Petal Width(cm)",
    "Petal Class",
]
print(iris_df)

# statistic summaries using numpy
iris_stats = iris_df.drop("Petal Class", axis=1)
mean_all = np.mean(iris_stats)
minimum = np.min(iris_stats)
maximum = np.max(iris_stats)
print(f"Mean: \n {mean_all}")
print(f"Min: \n {minimum}")
print(f"Max: \n {maximum}")


# five different plots
iris_type = plt.pie(iris_df, "Petal Class")
iris_type.show()

p_length = plt.violin(iris_df, "Petal Length(cm)", "Petal Class")
p_length.show()

p_width = plt.scatter(iris_df, "Petal Width(cm)", "Petal Class")
p_width.show()

s_length = plt.histogram(iris_df, "Sepal Length(cm)", "Petal Class")
s_length.show()

s_width = plt.box(iris_df, "Sepal Width(cm)", "Petal Class")
s_width.show()


# scikit-learn
scaler = StandardScaler().fit(iris_stats)
y = iris_df["Petal Class"]
X = scaler.transform(iris_stats)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
print(scaler.mean_)


# random forest
rand_tree = RandomForestClassifier(n_estimators=10)
rand_tree.fit(X_train, y_train)

print(rand_tree.predict(X_test))

tree_score = rand_tree.score(X_test, y_test)
print(f"tree score {tree_score}")

# OneVrestClassifier

classifier_d = OneVsOneClassifier(LinearSVC(random_state=0))
classifier_d.fit(X_train, y_train)
print(f"The best class label for each sample is: {classifier_d.predict(X_test)}")
print(f"Mean accuracy: {classifier_d.score(X_test, y_test)}")


# Pipeline

X_train, X_test, y_train, y_test = train_test_split(iris_stats, y, random_state=42)

rt_pipeline = Pipeline(
    [("scaler", StandardScaler()), ("rand_tree", RandomForestClassifier())]
)

rt_pipeline.fit(X_train, y_train)
pipe_score = rt_pipeline.score(X_test, y_test)
pipe_predict = rt_pipeline.predict(X_test)
print(f"Predict: {pipe_predict}")
print(f"Score: {pipe_score}")
