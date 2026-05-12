package lab8;

import java.util.Observable;

@SuppressWarnings("deprecation")
public class ExchangeRatio extends Observable {
    private final Currency currency;
    private double ratio;

    public ExchangeRatio(Currency currency, double ratio) {
        this.currency = currency;
        this.ratio = ratio;
    }

    public Currency getCurrency() {
        return currency;
    }

    public double getRatio() {
        return ratio;
    }

    public void setRatio(double ratio) {
        this.ratio = ratio;
        setChanged();
        notifyObservers(ratio);
    }

    @Override
    public String toString() {
        return String.format("Kurs %s: %.4f", currency, ratio);
    }
}
