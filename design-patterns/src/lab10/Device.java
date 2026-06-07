package lab10;

/**
 * Implementor wzorca Bridge.
 * Definiuje niskopoziomowy interfejs urządzeń, którym posługują się piloty (abstrakcja).
 */
public interface Device {
    boolean isEnabled();

    void enable();

    void disable();

    int getVolume();

    void setVolume(int percent);

    int getChannel();

    void setChannel(int channel);
}
