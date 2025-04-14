fillter_vector <- function(vector, b){
  if(length(b) != 1){
    return("Błąd: drugi argument musi byc skalarem(liczba całkowitą).")
  }
  if (b %% 1 != 0){
    return("Błąd: drugi argument musi być liczbą całkowitą.")
  }
  
  vector[vector %% b == 0]
}

vector <- 0:20
fillter_vector(vector, 3.2)
fillter_vector(vector, c(2, 2))
fillter_vector(vector, 2)


first_three <- function(vector){
  if(length(vector) < 3){
    return("Błąd: wektor jest za krótki – musi zawierać co najmniej 3 liczby.")
  }
  sort(vector)[1:3]
}


first_three(1:20)
first_three(c(1, 2))



frst_last_three <- function(vector){
  if(length(vector) < 3){
    return("wektor za krótki minimalna dlugość 3")
  }
  vector_sroted <- sort(vector)
  min_three <- vector_sroted[1:3]
  last_three <- tail(vector_sroted, 3)
  
  list(najmniejsze=min_three,
       najwieksze=last_three)
  c(min_three, last_three)
}

frst_last_three(c(1,2))
frst_last_three(1:20)



fibonnaci <- function(n){
  if(length(n) != 1){
    stop("argumentem musi być skalar")
  }
  if(n %% 1 != 0){
    stop("argumentem nie może być liczbą zmiennoprzecinkową
         nieutożsamianą z liczbą naturalną,")
  }
  if(n <= 0){
    stop("argumentem musi być liczba naturalna dodatnia")
  }
  if(n == 1){
    return(0)
  }
  if(n == 2){
    return(1)
  }
  return(fibonnaci(n-1) + fibonnaci(n-2))
}
fibonnaci(3)
fibonnaci(c(1,2))
fibonnaci(-1)
fibonnaci(4.3)

