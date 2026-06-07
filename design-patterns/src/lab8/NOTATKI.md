# Lab8 — System walutowy (Singleton + Observer)

## O co chodzi w jednym zdaniu
Masz **torby z pieniędzmi** w różnych walutach. Gdy zmienia się kurs walut,
wartości w torbach **same się przeliczają** — bez ręcznego odświeżania.

## Dwa wzorce naraz
- **Singleton** — istnieje tylko **jedna** tablica kursów w całym programie (`CurrencyRatios`).
- **Observer (Obserwator)** — paczki pieniędzy „nasłuchują" zmian kursów i automatycznie się aktualizują.

## Pliki — co robi każdy
| Plik | Rola | Najważniejsze |
|------|------|---------------|
| `Currency.java` | enum walut | USD, PLN, EUR, GBP, CHF |
| `ExchangeRatio.java` | jeden kurs (np. PLN = 0.22), jest **obserwowany** | `setRatio()` → `setChanged()` + `notifyObservers()` |
| `CurrencyRatios.java` | centralna tablica kursów + waluta wspólna, **Singleton**, też obserwowana | `getInstance()`, `setBaseCurrency()`, `refreshAll()` |
| `MoneyPacket.java` | paczka kasy (np. 100 PLN), jest **obserwatorem** | rejestruje się na 2 obserwowanych; `update()` → `updateValue()` |
| `BagFullOfMoney.java` | torba = lista paczek + suma | `getValue()` sumuje paczki |
| `Main.java` | demonstracja 3 zdarzeń | — |

## Mechanizm (najważniejsze!)
Zmieniasz kurs → obiekt woła `notifyObservers()` → każda `MoneyPacket` dostaje `update()`
→ przelicza swoją wartość → torba przy sumowaniu pokazuje już aktualne kwoty.
**Nikt ręcznie nie aktualizuje paczek — one same reagują.**

### Singleton (skrót)
```java
private static CurrencyRatios instance;
private CurrencyRatios() { ... }            // prywatny konstruktor
public static CurrencyRatios getInstance() {
    if (instance == null) instance = new CurrencyRatios();
    return instance;                         // zawsze ta sama instancja
}
```

### Każda paczka obserwuje DWIE rzeczy (wymóg z PDF)
1. swój kurs (`ExchangeRatio`) → reaguje, gdy zmieni się TEN kurs
2. singleton `CurrencyRatios` → reaguje, gdy zmieni się wspólna waluta

## 3 zdarzenia z zadania
1. **Zmiana kursu jednej waluty** → `er.setRatio()` na jednym kursie → aktualizują się **tylko** paczki tej waluty.
2. **Zmiana kursu wspólnej waluty** → `refreshAll()` → aktualizują się **wszystkie** paczki.
3. **Zmiana wyboru wspólnej waluty** → `setBaseCurrency()` → aktualizują się wszystkie kursy i wszystkie paczki.

## Kluczowa rzecz na obronę
**Selektywność reakcji (zdarzenie 1) wynika z tego, KOGO paczka obserwuje — a nie ze sprawdzania
argumentu w `update()`.** Paczka PLN jest zapisana tylko na kursie PLN, więc zmiana kursu PLN
powiadamia tylko ją. Dlatego `update()` nie musi rozróżniać typu zdarzenia.

## Drobna uwaga (gdyby ktoś pytał)
Stringi `"BASE_CURRENCY_CHANGED"` / `"REFRESH_ALL"` przekazywane do `notifyObservers(...)`
**nie są wymagane przez zadanie i nigdzie nie są sprawdzane** — `update()` je ignoruje.
To tylko ozdobnik; kod działa tak samo bez nich.
