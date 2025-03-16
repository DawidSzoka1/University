library(tidyverse)
library(scales)
acsNew <- read_csv("http://www.jaredlander.com/data/acsNew.csv")

summary(acsNew)

acsNew |> 
  ggplot(aes(x = FamilyIncome)) +
  geom_histogram(fill = "blue", color = "black")
?geom_histogram
acsNew$
p <- acsNew |> ggplot(aes(x = FamilyIncome))
p <- p + geom_histogram(
  aes(y = after_stat(count)/sum(after_stat(count))),
  color = "brown",
  fill = "antiquewhite"
  ) + 
  labs(title = "tytul", x = "przychod rodziny w $", y = "odsetek przypadkow")

p
pscale <- p + scale_x_continuous(labels = dollar_format(prefix = "zl"))
pscale

p3 <- acsNew |> ggplot(aes(x= FamilyIncome)) +
  geom_histogram(aes(y = after_stat(count)/sum(after_stat(count)))
    ,color = "brown", fill = "antiquewhite") +
  labs(title = "tytul", x = "przychody", y = "procent")
p3

p4 <- p3 + scale_x_continuous(labels = dollar) + scale_y_continuous(labels = percent)
p4

p5 <- p4 + theme(axis.text.y = element_text(angle = -75, color="green"),
                 plot.title = element_text(hjust = 0.5),
                 axis.text.x = element_text(angle = 5))
p5
p4 + theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5, color="peru"))
