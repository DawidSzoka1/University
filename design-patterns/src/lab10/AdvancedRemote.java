package lab10;

/**
 * Uściślona abstrakcja (RefinedAbstraction).
 * Pilot zaawansowany rozszerza zwykłego pilota o dodatkową funkcję wyciszania.
 */
public class AdvancedRemote extends Remote {

    public AdvancedRemote(Device device) {
        super(device);
    }

    /** Wyciszenie urządzenia poprzez ustawienie głośności na 0. */
    public void mute() {
        device.setVolume(0);
    }
}
