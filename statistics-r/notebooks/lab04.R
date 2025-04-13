fillter_vector <- function(vector, b){
  if(length(b) != 1){
    return("Błąd: drugi argument musi byc skalarem(liczba całkowitą).")
  }
  if (b %% 1 != 0){
    return("Błąd: drugi argument musi być liczbą całkowitą.")
  }
  
  vector[vector %% b == 0]
}

vector <- c(2, 3, 45,5654, 43, 32, 342)
fillter_vector(vector, 3.2)
fillter_vector(vector, c(2, 2))
fillter_vector(vector, 2)


first_three <- function(vector){
  if(length(vector) < 3){
    return("Błąd: wektor jest za krótki – musi zawierać co najmniej 3 liczby.")
  }
  sort(vector)[1:3]
}

first_three(vector)
first_three(c(1, 2))



