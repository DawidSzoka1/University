import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay


dataset = pd.read_csv('../docs/diabetes.csv')
#print(dataset) # Jeśli chce się obejrzeć zbiór
# Wyodrębnienie atrybutów warunkowych i atrybutu decyzyjnego
X = dataset.drop('Outcome', axis=1) # Cechy (Features, atrybuty warunkowe)
y = dataset['Outcome'] # Atrybut decyzyjny (Target
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.4, random_state=42
)
clf = svm.SVC(kernel='linear').fit(X_train, y_train) # Liniowa wersja SVM
# Użycie wytrenowanego klasyfikatora na zbiorze testowym (predykcja)
y_pred = clf.predict(X_test)
# Podanie dokładności predykcji
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Wyświetlenie macierzy pomyłek w wersji graficznej
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
display_labels=clf.classes_)
disp.plot()
roc_disp = RocCurveDisplay.from_estimator(clf, X_test, y_test)
plt.show()
