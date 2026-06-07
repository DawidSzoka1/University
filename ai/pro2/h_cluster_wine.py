import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram

# Wczytanie zbioru wine
names = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
         'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
         'proanthocyanins', 'color_intensity', 'hue',
         'od280_od315_of_diluted_wines', 'proline', 'class']
dataset = pd.read_csv('wine.csv', names=names)

# Usunięcie atrybutu decyzyjnego
X = dataset.drop('class', axis=1)

# Wykrywanie procentu brakujących wartości dowolnego rodzaju w kolumnach
print(X.isna().mean() * 100)

# Usunięcie wierszy z brakującymi wartościami
X = X.dropna()

# Standaryzacja wartości atrybutów
data_scaler = StandardScaler()
scaled_data = data_scaler.fit_transform(X)
print(scaled_data)

# Grupowanie obiektów z wykorzystaniem 4 wariantów algorytmu (różne metody
# liczenia odległości między skupieniami)
complete_clustering = linkage(scaled_data, method="complete", metric="euclidean")
average_clustering = linkage(scaled_data, method="average", metric="euclidean")
single_clustering = linkage(scaled_data, method="single", metric="euclidean")
ward_clustering = linkage(scaled_data, method="ward", metric="euclidean")

# Utworzenie i wyświetlenie dendrogramów
dendrogram(complete_clustering)
plt.title("Dendrogram (complete linkage method)")
plt.show()

dendrogram(average_clustering)
plt.title("Dendrogram (average linkage method)")
plt.show()

dendrogram(single_clustering)
plt.title("Dendrogram (single linkage method)")
plt.show()

dendrogram(ward_clustering)
plt.title("Dendrogram (Ward method)")
plt.show()