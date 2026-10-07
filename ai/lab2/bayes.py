import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics #Importowanie z scikit-learn modułu z metrykami
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve

dataset = pd.read_csv('../docs/diabetes.csv')
#print(dataset) # Jeśli chce się obejrzeć zbiór
# Wyodrębnienie atrybutów warunkowych i atrybutu decyzyjnego
X = dataset.drop('Outcome', axis=1) # Cechy (atrybuty warunkowe, Features)
y = dataset['Outcome'] # Atrybut decyzyjny (Target
# Podział zbioru uczącego na część treningową i część testową
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.4, random_state=42
)
clf = GaussianNB().fit(X_train, y_train)
# Użycie wytrenowanego klasy
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Balanced accuracy:",metrics.balanced_accuracy_score(y_test, y_pred))
print("F1:",metrics.f1_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Wyświetlenie macierzy pomyłek w wersji graficznej
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
display_labels=clf.classes_)
disp.plot()
#plt.show()
roc_disp = RocCurveDisplay.from_estimator(clf, X_test, y_test)
plt.show()
y_prob_test = clf.predict_proba(X_test)[:,1]
prec, recall, _ = precision_recall_curve(y_test, y_prob_test,
pos_label=clf.classes_[1])
pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()
plt.title("Krzywa PR (Precision-Recall)")
plt.show()
