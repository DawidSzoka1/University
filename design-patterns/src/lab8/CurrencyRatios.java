package lab8;

import java.util.HashMap;
import java.util.Map;
import java.util.Observable;

@SuppressWarnings("deprecation")
public class CurrencyRatios extends Observable {
    private static CurrencyRatios instance;
    private final Map<Currency, ExchangeRatio> ratios;
    private Currency baseCurrency;

    private CurrencyRatios() {
        ratios = new HashMap<>();
        baseCurrency = Currency.EUR;
    }

    public static CurrencyRatios getInstance() {
        if (instance == null) {
            instance = new CurrencyRatios();
        }
        return instance;
    }

    public void addRatio(Currency currency, double ratio) {
        ratios.put(currency, new ExchangeRatio(currency, ratio));
    }

    public ExchangeRatio getExchangeRatio(Currency currency) {
        return ratios.get(currency);
    }

    public Currency getBaseCurrency() {
        return baseCurrency;
    }

    public void setBaseCurrency(Currency baseCurrency, Map<Currency, Double> newRatios) {
        this.baseCurrency = baseCurrency;
        for (Map.Entry<Currency, Double> entry : newRatios.entrySet()) {
            ExchangeRatio er = ratios.get(entry.getKey());
            if (er != null) {
                er.setRatio(entry.getValue());
            }
        }
        setChanged();
        notifyObservers("BASE_CURRENCY_CHANGED");
    }

    public void refreshAll() {
        setChanged();
        notifyObservers("REFRESH_ALL");
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Wspólna waluta: ").append(baseCurrency).append("\nKursy:\n");
        for (ExchangeRatio er : ratios.values()) {
            sb.append("  ").append(er).append("\n");
        }
        return sb.toString();
    }
}
