library(tidyverse)
library(moments)

set.seed(20250322)
x <- rnorm(1000, mean = 7, sd = 2)

moment(x, order = 2)
moment(x, order = 2, absolute = TRUE)
moment(x, order = 2, central = TRUE)
moment(x, order = 2, central = TRUE, absolute = TRUE)

moment(x, order = 3)
moment(x, order = 3, absolute = TRUE)
moment(x, order = 3, central = TRUE)
moment(x,  order = 3, central = TRUE, absolute = TRUE)

help("all.moments")
all.moments(x, order.max = 3)
all.moments(x, order.max = 3, absolute = TRUE)
all.moments(x, order.max = 3, central = TRUE)
all.moments(x, order.max = 3, central = TRUE, absolute = TRUE)
