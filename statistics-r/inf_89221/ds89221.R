require(tidyverse)
set.seed(2024)
wektor <- sample(1:3, size = 20, replace=T,prob = c(0.1, 0.3, 0.6))
wektor

x <- 1515
if(x %% 2 == 0){
  "liczba parzysta"
}else{
  "liczba nieparzysta"
}

(3:333)[c(TRUE, FALSE, FALSE)]
seq(3, 333, 3)
for(i in 3:333){
  if(i %% 3 == 0){
    print(i)
  }
}

df <- data.frame(
  id = 1:10,
  kol2 = sample(1:10, size=10),
  kol3 = sample(10:20, size=10, replace = T)
)
df
suma <- 0
suma_kol3 <- 0
for(i in 1:nrow(df)){
  suma <- suma + df$kol2[i]
  suma_kol3 <- suma_kol3 + df$kol3[i]
}
suma / nrow(df)
suma_kol3 / nrow(df)
mean(df$kol2)
mean(df$kol3)


acsNew <- read_csv("http://www.jaredlander.com/data/acsNew.csv")
summary(acsNew)
