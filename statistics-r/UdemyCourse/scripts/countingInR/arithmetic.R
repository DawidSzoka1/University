x <- 18
x
y <- 6

z <- x + y
z
x - y
x*y
x / y
# reszta z dzielenia 
x %% y

# dzielenie calkowite
x %/% y

x == x %% y + y * (x %/% y)
# podnoszenie do potegi
x ^ y
#Dziala +0 i -0 co da albo inf albo -inf
y <- -0

# dziala i daje inf 
x / y

# Nan
x %% y

# inf
x %/% y

x > y
x <= y
x != y
x / y == x / y

#Nan == Nan nie jest porownywalne
x %% y == x %% y

