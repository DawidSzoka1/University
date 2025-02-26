
#ksztalt punktu pch="" lub pch=2
plot(x=1:nrow(chickwts), y=chickwts$weight, pch="+")
plot(x=1:nrow(chickwts), y=chickwts$weight, pch=15)
plot(x=1:nrow(chickwts), y=chickwts$weight, pch="-")
plot(x=1:nrow(chickwts), y=chickwts$weight, pch=23)

#Wielkosc punktow to jest cex=
plot(x=1:nrow(chickwts), y=chickwts$weight, cex=5)

#col="" okresla kolor obramowki
plot(x=1:nrow(chickwts), y=chickwts$weight, col="blue")
#mozna uzywac rgb bg="" kolor wypelnieni wypelnienie
plot(x=1:nrow(chickwts), y=chickwts$weight, pch=23, cex=4,col="blue",bg="#00FF00")

# opisywanie label xlab, ylab
plot(x=1:nrow(chickwts), y=chickwts$weight, pch="+", xlab="chicke", ylab="ciezar")

#nadanie konkretnemu rodzajowi pokarmu kolor i innny ksztalt
my_colors <- c('red', 'yellow', 'green', 'orange', 'violet', 'blue')
my_shapes <- (15:20)
color_column <- my_colors[as.numeric(chickwts$feed)]
shapes_column <- my_shapes[as.numeric(chickwts$feed)]

plot(x=1:nrow(chickwts), y=chickwts$weight, col=color_column, pch=shapes_column)

#legenda do plota
legend("topright", legend = levels(chickwts$feed),
       col = my_colors,
       pch = my_shapes
       )


#wykres slupkowy data to z jakich danych a height to wysokosc slupka od czego ma zalezyc
barplot(data=chickwts, height = chickwts$weight)

barplot(data = chickwts, height = chickwts$weight, col=color_column, pch=shapes_column)

#posortowanie wedlug wagi
ord_chick <- chickwts[order(chickwts$weight),]

hist(ord_chick$weight, breaks=10)

#ylim to limi osi y i musi to byc wektor dwu elementowy 
hist(ord_chick$weight, breaks=10, ylim=c(0,10), col="green")

boxplot(weight ~ feed, data= chickwts, varwidth = TRUE, notch=TRUE, col='grey')

#nalozenie jakies funkcji na grupe danych
?tapply

feed_mean <- tapply(chickwts$weight, chickwts$feed, FUN = mean)
feed_mean

barplot(feed_mean, main='Glowny opis plota', xlab='rodzaj paszy',
        ylab='weight', col=my_colors, horiz=TRUE)

#pare wykresow na raz na ekranie mfrow=c(2, 2) to bede wiersze i kolumny podzielenie i tyle
# bedzie obrazkow moglo sie zmiescic 
par()
par(mfrow=c(2,2))
#reset ustawien
dev.off()

#nastepne wykonanie obrazu zapisze do pliku graficznego ten obraz
png('', width=500, height=350, res=72)

#potem trzeba wyczyscic to
dev.off()

