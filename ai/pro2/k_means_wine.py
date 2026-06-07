import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import seaborn as sns

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

# Skalowanie danych (ważne dla KMeans!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA do redukcji wymiarów do 2D, żeby móc narysować skupienia
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Wyznaczanie optymalnej liczby skupień k metodą Silhouette score
K = range(2, 8)
fits = []
score = []

for k in K:
    # Utworzenie modelu metodą k-średnich w postaci rodziny skupień obiektów zbioru X_pca
    model = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(X_pca)
    # Dodanie modelu do fits
    fits.append(model)
    # Wyznaczenie wartości silhouette score dla utworzonego modelu
    score.append(silhouette_score(X_pca, model.labels_, metric='euclidean'))

print(score)

# Utworzenie wykresu
sns.lineplot(x=K, y=score).set(title='Wykres Silhouette score -- Wine',
                               xlabel='Liczba skupień (k)', ylabel='Silhouette score')
plt.show()

# Wyznaczanie optymalnej liczby skupień k metodą "łokcia" (Elbow)
# Obliczanie WCSS (inercja) dla różnej liczby skupień (k od 2 do 7)
wcss = []
for i in range(2, 8):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300,
                    n_init=10, random_state=0)
    kmeans.fit(X_pca)
    # kmeans.inertia_ zwraca sumę kwadratów odległości od środka skupienia
    wcss.append(kmeans.inertia_)

# Wykreślenie wykresu dla metody łokcia
plt.plot(range(2, 8), wcss)
plt.title('Wykres dla metody "łokcia" -- Wine')
plt.xlabel('Liczba skupień (k)')
plt.ylabel('WCSS (inercja)')
plt.grid(True)
plt.show()

# Wykreślenie skupień dla wybranych k (np. 2 i 3)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=fits[0].labels_)
plt.title("Rozkład obiektów (PCA) dla k=2")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()

sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=fits[1].labels_)
plt.title("Rozkład obiektów (PCA) dla k=3")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()

# Zastosowanie k-means na zbiorze X_pca dla k=3
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_pca)

# Przypisanie etykiet skupień do obiektów (obserwacji)
labels = kmeans.labels_
print(labels)