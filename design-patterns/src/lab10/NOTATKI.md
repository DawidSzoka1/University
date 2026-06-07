# Lab10 — Piloty i urządzenia (Bridge — most)

## O co chodzi w jednym zdaniu
Masz **piloty** (zwykły, zaawansowany) i **urządzenia** (TV, radio, DVD).
Każdy pilot może obsługiwać **dowolne** urządzenie i można je **przepinać w locie** —
piloty i urządzenia rozwijają się **niezależnie od siebie**.

## Wzorzec
**Bridge (most)** — oddziela **abstrakcję** (pilot) od **implementacji** (urządzenie),
żeby obie strony mogły zmieniać się niezależnie. Pilot trzyma **referencję** na urządzenie
(`device`) i **oddelegowuje** do niego polecenia.

```
  Pilot (abstrakcja)  --- device --->  Device (implementor)
  Remote                                 TV / Radio / DvdPlayer
  AdvancedRemote
```

## Pliki — co robi każdy
| Plik | Strona mostu | Najważniejsze |
|------|--------------|---------------|
| `Device.java` | **Implementor** (interfejs) | `enable/disable/isEnabled`, `get/setVolume`, `get/setChannel` |
| `BaseDevice.java` | wspólna baza urządzeń | trzyma zmienne: `enabled`, `muted`, `volume`, `channel` |
| `Tv.java` | konkretne urządzenie | cecha unikalna: **obraz** (`togglePicture`, format) |
| `Radio.java` | konkretne urządzenie | cecha unikalna: **zapisane częstotliwości** (`saveFrequency`) |
| `DvdPlayer.java` | **dodatkowe** urządzenie | cecha unikalna: **odtwarzanie** (`insertDisc/play/stop`) |
| `Remote.java` | **Abstrakcja** (pilot) | trzyma `device`; `togglePower`, `volumeUp/Down`, `channelUp/Down`, `setDevice()` |
| `AdvancedRemote.java` | rozszerzona abstrakcja | dodaje `mute()` → `device.setVolume(0)` |
| `Main.java` | klient | pokazuje przepinanie pilota między urządzeniami |

## Mechanizm (najważniejsze!)
Pilot nie wie, jak działa urządzenie — tylko **woła jego metody**:
```java
public void togglePower() {
    if (device.isEnabled()) device.disable();
    else device.enable();
}
public void channelUp() {
    int old = device.getChannel();
    device.setChannel(old + 1);
}
```
A `setDevice(Device dd)` **paruje pilota z nowym urządzeniem** (jak przyciski wyboru
na pilocie uniwersalnym) — ten sam pilot obsługuje raz radio, raz DVD.

## Zmienne na właściwym poziomie (wymóg)
- Wspólne (`enabled`, `volume`, `channel`, `muted`) → w `BaseDevice`.
- Unikalne → w konkretnych klasach: TV ma obraz, Radio ma częstotliwości, DVD ma odtwarzanie.

## Ważny niuans z zadania
**Pilot nie wie, czy operacja ma sens.** Wysyła `volumeDown()` nawet gdy urządzenie jest
wyciszone (`muted`), albo `volumeUp()` gdy jest wyłączone (`enabled == false`).
To celowe — pilot tylko wysyła komunikaty, urządzenie samo decyduje co z nimi zrobić.

## Kluczowa rzecz na obronę
**Bez Bridge potrzebowalibyśmy klas typu „PilotDoTV", „PilotZaawansowanyDoRadia" — eksplozja
kombinacji.** Bridge rozdziela to na dwie niezależne hierarchie (piloty × urządzenia),
połączone jedną referencją `device`. Nowy pilot albo nowe urządzenie dodajesz osobno,
bez ruszania drugiej strony.
