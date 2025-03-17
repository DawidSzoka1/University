library(tidyverse)
library(forcats)
acsNew <- read_csv("http://www.jaredlander.com/data/acsNew.csv")

acsNew |> ggplot(aes(x = Language, fill = Language)) + 
  geom_bar(width = 0.5) +
  labs(
    title="Rozkład języków w badaniach",
    fill = "Języki",
    caption = "dane z jaredlander",
    x="Język",
    y="ilość") +
  theme_bw()



acsNew |> ggplot(aes(x = fct_rev(fct_infreq(Language)), fill = Language)) + 
  geom_bar(width = 0.5) +
  labs(
    title="Rozklad jezykow w badaniach",
    fill = "Jezyki",
    caption = "dane z jaredlander",
    x="Jezyk",
    y="ilosc") +
  theme_bw() +
  theme(axis.ticks.x = element_blank())

acsNew |> ggplot(aes(x = fct_infreq(Language), fill = Language)) + 
  geom_bar(width = 0.5) +
  labs(
    title="Rozklad jezykow w badaniach",
    fill = "Jezyki",
    caption = "dane z jaredlander",
    x="Jezyk",
    y="ilosc") +
  theme_bw() +
  theme(axis.ticks.x = element_blank())


acsNew |> 
  ggplot(aes(x = HeatingFuel, y = HouseCosts, fill = HeatingFuel)) +
  geom_violin() + 
  labs(x = "rodzaj paliwa", 
       title = "Wykres ceny domu do rodzaju paliwa grzewczego",
       y="cena domu",
        fill="rodzaje paliwa grzewczego")


str(acsNew)
acsNew <- acsNew |> mutate(across(where(is.character), as.factor))      
summary(acsNew)           

factors_vars <- acsNew |> select(where(is.factor)) |> names()

factors_vars


plots2 <- map(factors_vars, function(var) acsNew |> 
                ggplot(aes(x = .data[[var]], fill = .data[[var]])) +
                geom_bar())
plots2[4]



acsNew |> ggplot(aes(x =FamilyIncome , y= Insurance)) +
  geom_point(color="red", alpha=0.6)
acsNew |> ggplot(aes(x = HouseCosts, y = Insurance)) +
  geom_point(color="red")
  