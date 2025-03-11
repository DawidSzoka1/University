#DLugosc napisu
nchar(title)

substr(title, 1, nchar(title) - 2)

toupper(title)

w <- strsplit(title, ' ')
w[1]
w <- c('pier f', 'drugi f', 'trzeci f')
paste(w[1], w[2], w[3], sep='-')

sub('Test', 'Nie test', title)

grep('to', title)
grep('f', w, value = TRUE)
#Grep logiczne czyli zwraca TRUE albo FALSE
grepl('f', w)

result <- format(123.264234, digits = 4)
result

result <- format(123, nsmall = 4)
result

multiline_text <- 'first line\nsecond line\nthird line'
nchar(multiline_text)
poem <- 'ona wyjechala juz spij sam'

new_poem <- paste(substr(poem, 1, 8), substr(poem, 20, 26))
toupper(new_poem)
nonsens <- c("Szedł facet koło bajora i go przymuliło. ", "Szedł facet koło bałaganu i zaczął gadać bez sensu. ", "Szedł facet koło betoniarki i się zmieszał. ", 
             "Szedł facet koło butelki i mu wlali. ", "Szedł facet koło cementu i go zamurowało. ", 
             "Szedł facet koło cukierni i coś go wpierniczyło. ", "Szedł facet koło czołgu i go oblazły gąsienice. ",
             "Szedł facet koło dołu i się poniżył. ", "Szedł facet koło drutów i go zelektryzowała wiadomość. ", 
             "Szedł facet koło dźwigu i się uniósł. ", "Szedł facet koło gilotyny i stracił głowę. ", 
             "Szedł facet koło haka i ktoś się do niego przyczepił. ", "Szedł facet koło koparki i dał się nabrać. ", 
             "Szedł facet koło korka i się zatkał. ", "Szedł facet koło kościoła i łupnęło go w krzyżu. ", 
             "Szedł facet koło kranu i go olali. ", "Szedł facet koło latarni i go oświeciło. ", 
             "Szedł facet koło lustra i mu odbiło. ", "Szedł facet koło łopaty i go wkopali. ", 
             "Szedł facet koło młotka i był trochę stuknięty. ", "Szedł facet koło mydła i zaczął się pienić. ", 
             "Szedł facet koło noża i zarżnął kawał. ", "Szedł facet koło piasku i go wsypali. ", 
             "Szedł facet koło pieca i się zapalił. ", "Szedł facet koło piły i się urżnął. ", 
             "Szedł facet koło pomnika i skamieniał. ", "Szedł facet koło półki z przyprawami i go opieprzyli. ", 
             "Szedł facet koło prysznica i się spłukał. ", "Szedł facet koło punktu skupu opakowań szklanych i nabili go w butelkę. ", 
             "Szedł facet koło reflektora i go olśniło. ", "Szedł facet koło rzeki i mu zmyli głowę. ", 
             "Szedł facet koło samochodu i się przejechał. ", "Szedł facet koło saperki i mu dokopali. ", 
             "Szedł facet koło sklepu rybnego i go wyśledzili. ", "Szedł facet koło spłuczki i się spuścił. ", 
             "Szedł facet koło stadniny i zrobili go w konia. ", "Szedł facet koło szczoty i go przeczyściło. ", 
             "Szedł facet koło sznurka i oberwał. ", "Szedł facet koło śruby i się gdzieś wkręcił. ", 
             "Szedł facet koło topora katowskiego i go z nóg ścięło. ", "Szedł facet koło walca i zaczął się płaszczyć. ",
             "Szedł facet koło więzienia i się zamknął. ", "Szedł facet koło wodociągu i się zmył. ", 
             "Szedł facet koło zlewu i go zatkało. ", "Szedł facet po torach i się wykoleił. ", 
             "Szedł facet po zboczu i się stoczył. ", "Szedł facet podczas wichury i gdzieś zwiał. ", 
             "Szedł facet przez budowę i go zamurowało. ", "Szedł facet przez las i zdębiał. ", 
             "Szedł facet przez lód i się załamał. ", "Szedł facet przez ogród i nalał w pory.")
grep("koło", nonsens)
grepl("koło", nonsens)
nonsens[grep("koło", nonsens)]
nonsens[grep("i go", nonsens)]

message <- format("The end", width = 20, justify = 'c')
message
