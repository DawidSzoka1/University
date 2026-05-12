package lab8;

import java.util.Observable;
import java.util.Observer;

@SuppressWarnings("deprecation")
public class MoneyPacket implements Observer {
    private final double originalAmount;
    private final Currency originalCurrency;
    private double value;

    public MoneyPacket(double originalAmount, Currency originalCurrency) {
        this.originalAmount = originalAmount;
        this.originalCurrency = originalCurrency;
        
        // Register as observer
        CurrencyRatios.getInstance().addObserver(this);
        ExchangeRatio er = CurrencyRatios.getInstance().getExchangeRatio(originalCurrency);
        if (er != null) {
            er.addObserver(this);
        }
        
        updateValue();
    }

    private void updateValue() {
        CurrencyRatios cr = CurrencyRatios.getInstance();
        if (originalCurrency == cr.getBaseCurrency()) {
            this.value = originalAmount;
        } else {
            ExchangeRatio er = cr.getExchangeRatio(originalCurrency);
            if (er != null) {
                this.value = originalAmount * er.getRatio();
            } else {
                this.value = 0.0; // Or handle error
            }
        }
    }

    @Override
    public void update(Observable o, Object arg) {
        updateValue();
    }

    public double getValue() {
        return value;
    }

    @Override
    public String toString() {
        return String.format("%.2f %s (Wartość: %.2f %s)", 
            originalAmount, originalCurrency, value, CurrencyRatios.getInstance().getBaseCurrency());
    }
}
