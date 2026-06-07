package lab10;

/**
 * Abstrakcja wzorca Bridge.
 * Przechowuje referencję na urządzenie (Implementor) i oddelegowuje do niego żądania.
 * Pilot jest uniwersalny - może obsługiwać dowolne urządzenie i jest z nim sparowany
 * przez zmienną device. Pilot nie ma wiedzy, czy operacja ma sens
 * (np. wysyła volumeDown() mimo że urządzenie jest wyciszone lub wyłączone).
 */
public class Remote {
    protected Device device;

    public Remote(Device device) {
        this.device = device;
    }

    /** Parowanie pilota z podanym urządzeniem (przyciski wyboru urządzenia). */
    public void setDevice(Device dd) {
        this.device = dd;
        System.out.println("Pilot: sparowano z nowym urządzeniem.");
    }

    public void togglePower() {
        if (device.isEnabled()) {
            device.disable();
        } else {
            device.enable();
        }
    }

    public void volumeDown() {
        device.setVolume(device.getVolume() - 10);
    }

    public void volumeUp() {
        device.setVolume(device.getVolume() + 10);
    }

    public void channelDown() {
        int old = device.getChannel();
        device.setChannel(old - 1);
    }

    public void channelUp() {
        int old = device.getChannel();
        device.setChannel(old + 1);
    }
}
