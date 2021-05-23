import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve
from sklearn.model_selection import cross_val_score
import seaborn as sns
from pandas.plotting import parallel_coordinates, radviz
import plotly.graph_objects as go



dataset = pd.read_excel('iris.xls')
X = dataset.iloc[:,:4].values
y = dataset.iloc[:,4].values

#sum_X = X.isnull().sum()
#sum_y = y.isnull().sum()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('Logistic Regression')
print(cm)
log_accuracy = accuracy_score(y_test, y_pred)
print(log_accuracy)

accuracies = cross_val_score(estimator = log_reg, X = X_train, y = y_train, cv = 10)
log_std = accuracies.std()*100




knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('KNN')
print(cm)
knn_accuracy = accuracy_score(y_test, y_pred)
print(knn_accuracy)
#print(classification_report(y_test, y_pred, output_dict=True))

accuracies = cross_val_score(estimator = knn, X = X_train, y = y_train, cv = 10)
knn_std = accuracies.std()*100




svm = SVC(kernel = 'rbf', random_state = 0)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('SVM')
print(cm)
svm_accuracy = accuracy_score(y_test, y_pred)
print(svm_accuracy)

accuracies = cross_val_score(estimator = svm, X = X_train, y = y_train, cv = 10)
svm_std = accuracies.std()*100



naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)

y_pred = naive_bayes.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('Naive Bayes')
print(cm)
nvb_accuracy = accuracy_score(y_test, y_pred)
print(nvb_accuracy)

accuracies = cross_val_score(estimator = naive_bayes, X = X_train, y = y_train, cv = 10)
nvb_std = accuracies.std()*100



dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('DT')
print(cm)
dt_accuracy = accuracy_score(y_test, y_pred)
print(dt_accuracy)

accuracies = cross_val_score(estimator = dt, X = X_train, y = y_train, cv = 10)
dt_std = accuracies.std()*100



rf = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 0)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('RF')
print(cm)
rf_accuracy = accuracy_score(y_test, y_pred)
print(rf_accuracy)


accuracies = cross_val_score(estimator = rf, X = X_train, y = y_train, cv = 10)
rf_std = accuracies.std()*100



#Plotting Class
sns.set()
sns.pairplot(dataset[['sepal length', 'sepal width', 'petal length', 'petal width', 'iris']],
             hue="iris", diag_kind="kde")



pd.plotting.parallel_coordinates(dataset, 'iris')
plt.show()



sns.jointplot(x="sepal length", y="sepal width", data=dataset, height=10,ratio=10, kind='hex',color='red')
plt.show()



pd.plotting.radviz(dataset, 'iris')
plt.show()



fig = px.scatter_3d(dataset, x='sepal length', y='sepal width', z='petal width',
                    color='petal length', symbol='iris')
fig.show()
fig.write_html('scatter.html', auto_open=True)



le = LabelEncoder()
id = pd.DataFrame(le.fit_transform(y))
dataset = pd.concat([dataset, id], axis=1)

dataset_new = dataset.rename(columns={0: 'iris_id'})

fig = px.parallel_coordinates(dataset_new, color='iris_id',
                              dimensions=['sepal length', 'sepal width', 
                                          'petal length', 'petal width', "iris_id"],
                              color_continuous_scale=px.colors.diverging.Tealrose,
                              color_continuous_midpoint=2)
fig.show()
fig.write_html('Iris.html', auto_open=True)



labels = ["Logistic Regression", "KNN", "SVM", 
          "Naive Bayes", "Decision Tree", "Random Forest"]

fig = go.Figure()
fig.add_trace(go.Bar(
    x=labels,
    y=[log_accuracy, knn_accuracy, svm_accuracy, nvb_accuracy, dt_accuracy, rf_accuracy],
    name='Accuracy Scores',
    marker_color='indianred'
))
fig.add_trace(go.Bar(
    x=labels,
    y=[log_std, knn_std, svm_std, nvb_std, dt_std, rf_std],
    name='Standart Deviation',
    marker_color='lightsalmon'
))

fig.update_layout(barmode='group', xaxis_tickangle=-45)
fig.show()
fig.write_html('plot.html', auto_open=True)



#ROC Curves Plotting
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from itertools import cycle


le = LabelEncoder()
y_2 = le.fit_transform(y)

y_2 = label_binarize(y_2, classes=[0, 1, 2])
n_classes = y_2.shape[1]

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X, y_2, test_size=.33,
                                                    random_state=0)

random_state = np.random.RandomState(0)

#svm, log_reg 
classifier = OneVsRestClassifier(svm)
y_score = np.array(classifier.fit(X_train_2, y_train_2).decision_function(X_test_2))

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_2[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_test_2.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


colors = cycle(['blue', 'red', 'green'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=1.5, label='ROC curve of class {0} (area = {1:0.2f})' ''.format(i+1, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k-', lw=1.5)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class data')
plt.legend(loc="lower right")
plt.show()


    
    