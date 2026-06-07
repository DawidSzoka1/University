package lab10;

/**
 * Wspólna implementacja dla wszystkich urządzeń.
 * Przechowuje zmienne wspólne na właściwym poziomie dziedziczenia:
 * enabled, volume, channel oraz muted.
 */
public abstract class BaseDevice implements Device {
    protected static final int MIN_VOLUME = 0;
    protected static final int MAX_VOLUME = 100;

    protected boolean enabled = false;
    protected boolean muted = false;
    protected int volume = 30;
    protected int channel = 1;

    /** Nazwa urządzenia używana w komunikatach. */
    protected abstract String name();

    @Override
    public boolean isEnabled() {
        return enabled;
    }

    @Override
    public void enable() {
        enabled = true;
        System.out.println(name() + ": włączono urządzenie.");
    }

    @Override
    public void disable() {
        enabled = false;
        System.out.println(name() + ": wyłączono urządzenie.");
    }

    @Override
    public int getVolume() {
        return volume;
    }

    @Override
    public void setVolume(int percent) {
        if (percent < MIN_VOLUME) {
            percent = MIN_VOLUME;
        } else if (percent > MAX_VOLUME) {
            percent = MAX_VOLUME;
        }
        volume = percent;
        muted = (volume == MIN_VOLUME);
        System.out.println(name() + ": ustawiono głośność na " + volume + "%"
                + (muted ? " (wyciszono)" : "") + ".");
    }

    @Override
    public int getChannel() {
        return channel;
    }

    @Override
    public void setChannel(int channel) {
        this.channel = channel;
        System.out.println(name() + ": przełączono na kanał " + channel + ".");
    }

    public boolean isMuted() {
        return muted;
    }
}
