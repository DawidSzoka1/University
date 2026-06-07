# Lab9 — Gra przygodowa (Composite — drzewo lokacji)

## O co chodzi w jednym zdaniu
Tekstowa **gra przygodowa**: wędrujesz przez losowo wygenerowaną mapę lokacji
(pokoje, korytarze), aż dotrzesz do **Nirwany** (wygrana) albo **Czarnej Dziury** (przegrana).

## Wzorzec
**Composite** — mapa to **drzewo lokacji**. Każda lokacja (`Area`) to węzeł drzewa:
- jedne mają dzieci (`Room` → jedna następna lokacja, `Corridor` → lista drzwi),
- inne to liście kończące grę (`Nirvana`, `BlackHole`).

Cała gra to po prostu **chodzenie po tym drzewie**: każda lokacja po wejściu zwraca następną.

## Pliki — co robi każdy
| Plik | Rola | Najważniejsze |
|------|------|---------------|
| `Area.java` | **abstrakcyjna** lokacja (węzeł drzewa) | `abstract Area enter(player, scanner)` — zwraca następną lokację |
| `Room.java` | pokój: nagroda / bomba / pusty | po wejściu pyta gracza i wraca `next` (jedna następna lokacja) |
| `Corridor.java` | korytarz z kilkoma drzwiami | gracz wybiera drzwi (1..N), zwraca wybraną lokację |
| `Nirvana.java` | liść — **wygrana** | `player.win()`, zwraca `null` (koniec) |
| `BlackHole.java` | liść — **przegrana** | `player.lose()`, zwraca `null` (koniec) |
| `RoomContent.java` | enum zawartości pokoju | REWARD, BOMB, EMPTY |
| `MapGenerator.java` | **losowo buduje drzewo lokacji** (rekurencyjnie) | `generate()` → `buildArea(depth)` |
| `Player.java` | stan gracza | score, lives (3 życia ♥), won/gameOver |
| `HiScore.java` | TOP 5 wyników w pliku | `save()`, `load()`, `print()` (zapis do `lab9_hiscore.txt`) |
| `Game.java` | pętla gry | `start()` → `playRound()` |
| `Main.java` | uruchomienie | `new Game().start()` |

## Mechanizm (najważniejsze!)
Cała pętla gry to jedna linijka:
```java
while (current != null && !player.isGameOver()) {
    current = current.enter(player, scanner);   // wejdź do lokacji, dostań następną
}
```
- `enter()` zwraca **następną lokację** → idziemy dalej,
- `enter()` zwraca **null** → koniec gry (Nirwana / Czarna Dziura / brak żyć).

Dzięki temu, że każda lokacja sama wie, dokąd prowadzi, pętla nie musi nic wiedzieć
o rodzajach lokacji — to jest siła **polimorfizmu** w Composite.

## Jak powstaje mapa (MapGenerator)
Rekurencyjnie buduje drzewo w głąb (max głębokość 7):
- losowo robi `Room` albo `Corridor`,
- od głębokości 3 może pojawić się Nirwana/Czarna Dziura,
- na końcu (głębokość 7) zawsze liść (Nirwana albo Czarna Dziura).

## Kluczowa rzecz na obronę
**Wspólny interfejs `Area.enter()` sprawia, że pokój, korytarz i liście są obsługiwane
tak samo — pętla gry tylko podąża za zwracaną lokacją.** Lokacje-kontenery (Room, Corridor)
zawierają inne lokacje (dzieci), lokacje-liście (Nirvana, BlackHole) kończą grę. To jest istota Composite.
