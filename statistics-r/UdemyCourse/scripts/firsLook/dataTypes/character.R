x <- 16
typeof(x)

x <- integer(16)
typeof(x)

x <- TRUE
typeof(x)
typeof(c(1, 2, '3'))
x <- complex(real = 1, imaginary = -2)
typeof(x)

x <- Sys.time()
typeof(x)
x

title <- 'Test to jest'
title
print(title, quote = FALSE)
typeof(title)
is.character(title)
as.character(123)

`title of a book` <- 'long name variable'
`title of a book`
?Quotes

a_longer_text <- "\tCOl1\tCol2\tcol3\nRow1"
cat(a_longer_text)
a_longer_text
writeLines(a_longer_text)
message(a_longer_text)
print(a_longer_text)
print(NA, na.print = "-----")
