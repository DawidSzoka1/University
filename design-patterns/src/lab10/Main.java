package lab10;

/**
 * Klient wzorca Bridge.
 * Pokazuje niezależność abstrakcji (piloty) od implementacji (urządzenia):
 * ten sam pilot może obsługiwać różne urządzenia po przeparowaniu (setDevice).
 */
public class Main {
    public static void main(String[] args) {
        // Przykład 1: zwykły pilot sparowany z telewizorem
        Tv tv = new Tv();
        Remote remote = new Remote(tv);
        System.out.println("--- Zwykły pilot + telewizor ---");
        remote.togglePower();   // włączenie
        remote.volumeUp();
        remote.channelUp();
        tv.togglePicture();     // cecha unikalna TV
        tv.setPictureFormat("4:3");

        // Przykład 2: pilot zaawansowany sparowany z radiem
        Radio radio = new Radio();
        radio.saveFrequency(90.5);
        radio.saveFrequency(101.3);
        AdvancedRemote advanced = new AdvancedRemote(radio);
        System.out.println("\n--- Pilot zaawansowany + radio ---");
        advanced.togglePower();
        advanced.channelUp();   // strojenie na zapisaną częstotliwość
        advanced.volumeUp();
        advanced.mute();        // dodatkowa funkcja - wyciszenie

        // Pilot nie wie, czy operacja ma sens - obniża głośność mimo wyciszenia
        advanced.volumeDown();

        // Przykład 3: przeparowanie tego samego pilota z odtwarzaczem DVD
        DvdPlayer dvd = new DvdPlayer();
        dvd.insertDisc("Matrix");
        System.out.println("\n--- Ten sam pilot przeparowany na odtwarzacz DVD ---");
        advanced.setDevice(dvd);
        advanced.togglePower();
        dvd.play();             // cecha unikalna DVD
        advanced.channelUp();   // przejście do następnego rozdziału
        advanced.mute();
        dvd.stop();
        advanced.togglePower(); // wyłączenie

        // Pilot nie wie, że urządzenie jest wyłączone - i tak wysyła komunikaty
        advanced.volumeUp();
    }
}
