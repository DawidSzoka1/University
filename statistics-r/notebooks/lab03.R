library(tidyverse)
library(moments)
library(e1071)

set.seed(20250322)
x <- rnorm(1000, mean = 7, sd = 2)
# na.rm = TRUE usuwa braki danych 
moments::moment(x, order = 2, na.rm=TRUE)
moments::moment(x, order = 2, absolute = TRUE, na.rm=TRUE)
moments::moment(x, order = 2, central =  TRUE)
moments::moment(x, order = 2, central = TRUE, absolute = TRUE, na.rm=TRUE)

moments::moment(x, order = 3, na.rm=TRUE)
moments::moment(x, order = 3, absolute = TRUE, na.rm=TRUE)
moments::moment(x, order = 3, central = TRUE, na.rm=TRUE)
moments::moment(x,  order = 3, central = TRUE, absolute = TRUE, na.rm=TRUE)

help("all.moments")
all.moments(x, order.max = 3)
all.moments(x, order.max = 3, absolute = TRUE)
all.moments(x, order.max = 3, central = TRUE)
all.moments(x, order.max = 3, central = TRUE, absolute = TRUE)

R.version.string
