# Tworzenie macierzy danych
dane <- matrix(c(
  86, 31, 132, 19,
  17, 64, 43, 13,
  54, 39, 132, 33,
  30, 17, 37, 54
), nrow = 4, byrow = TRUE)

# Nadanie nazw wierszom i kolumnom
rownames(dane) <- c("Pieniądze", "Dzieci", "Zainteresowania", "Inne")
colnames(dane) <- c("Pieniądze", "Dzieci", "Zainteresowania", "Inne")

# Test chi-kwadrat
test <- chisq.test(dane)

# Wyniki testu
print(test)
print("Ponieważ wartość p (p-value) jest znacznie mniejsza niż przyjęty
      poziom istotności α = 0.05, odrzucamy hipotezę zerową. 
      Istnieje statystycznie istotna zależność między poglądami mężów
      i żon na temat przyczyn kryzysu w ich małżeństwach.")


tabela <- matrix(c(57, 18, 24, 91), nrow = 2, byrow = TRUE)
rownames(tabela) <- c("Nadwaga", "Brak nadwagi")
colnames(tabela) <- c("Ciśnienie ++", "Ciśnienie OK")

# Test chi-kwadrat
test <- chisq.test(tabela, correct = TRUE)  
print(test)
print("Ponieważ p-value < 0.05, odrzucamy hipotezę zerową.
      Istnieje statystycznie istotna zależność między nadwagą
      a ciśnieniem krwi – osoby z nadwagą częściej mają podwyższone ciśnienie.")