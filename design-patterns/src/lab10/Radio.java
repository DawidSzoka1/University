package lab10;

import java.util.ArrayList;
import java.util.List;

/**
 * Konkretny Implementor: radio.
 * Cecha unikalna: lista zapisanych częstotliwości (presety stacji).
 * Kanał interpretowany jest jako indeks zapisanej częstotliwości.
 */
public class Radio extends BaseDevice {
    private final List<Double> savedFrequencies = new ArrayList<>();

    @Override
    protected String name() {
        return "Radio";
    }

    /** Cecha unikalna dla radia - zapisanie częstotliwości stacji. */
    public void saveFrequency(double frequency) {
        savedFrequencies.add(frequency);
        System.out.println(name() + ": zapisano częstotliwość " + frequency + " MHz pod numerem "
                + savedFrequencies.size() + ".");
    }

    @Override
    public void setChannel(int channel) {
        super.setChannel(channel);
        if (channel >= 1 && channel <= savedFrequencies.size()) {
            System.out.println(name() + ": strojenie na " + savedFrequencies.get(channel - 1) + " MHz.");
        } else {
            System.out.println(name() + ": brak zapisanej częstotliwości pod numerem " + channel + ".");
        }
    }

    public List<Double> getSavedFrequencies() {
        return savedFrequencies;
    }
}
