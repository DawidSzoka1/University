library(tidyverse)
library(forcats)
acsNew <- read_csv("http://www.jaredlander.com/data/acsNew.csv")

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


acsNew |> ggplot(aes(y = HouseCosts, x = HeatingFuel, fill = HeatingFuel)) +
  geom_violin() 


str(acsNew)
acsNew <- acsNew |> mutate(across(where(is.character), as.factor))      
summary(acsNew)           

factors_vars <- acsNew |> select(where(is.factor)) |> names()

factors_vars


plots2 <- map(factor_vars, function(var) acsNew |> 
                ggplot(aes(x = .data[[var]], fill = .data[[var]])) +
                geom_bar())
plots2[4]
