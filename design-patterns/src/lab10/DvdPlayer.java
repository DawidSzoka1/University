package lab10;

/**
 * Dodatkowy konkretny Implementor: odtwarzacz DVD.
 * Obsługiwany w podobny sposób jak pozostałe urządzenia (głośność, kanał = ścieżka/rozdział).
 * Cecha unikalna: sterowanie odtwarzaniem (play/pause/stop, tytuł płyty).
 */
public class DvdPlayer extends BaseDevice {
    private boolean playing = false;
    private String disc = "brak płyty";

    @Override
    protected String name() {
        return "Odtwarzacz DVD";
    }

    /** Cecha unikalna - włożenie płyty. */
    public void insertDisc(String disc) {
        this.disc = disc;
        System.out.println(name() + ": włożono płytę \"" + disc + "\".");
    }

    /** Cecha unikalna - rozpoczęcie odtwarzania. */
    public void play() {
        playing = true;
        System.out.println(name() + ": odtwarzanie \"" + disc + "\".");
    }

    /** Cecha unikalna - zatrzymanie odtwarzania. */
    public void stop() {
        playing = false;
        System.out.println(name() + ": zatrzymano odtwarzanie.");
    }

    @Override
    public void setChannel(int channel) {
        this.channel = channel;
        System.out.println(name() + ": przejście do rozdziału " + channel + ".");
    }

    public boolean isPlaying() {
        return playing;
    }

    public String getDisc() {
        return disc;
    }
}
