package lab8;

import java.util.HashMap;
import java.util.Map;

public class Main {
    public static void main(String[] args) {
        CurrencyRatios ratios = CurrencyRatios.getInstance();
        
        // Initial setup (Base: EUR)
        ratios.addRatio(Currency.PLN, 0.22);
        ratios.addRatio(Currency.USD, 0.92);
        ratios.addRatio(Currency.GBP, 1.17);
        ratios.addRatio(Currency.CHF, 1.03);
        ratios.addRatio(Currency.EUR, 1.0);

        BagFullOfMoney bag1 = new BagFullOfMoney();
        bag1.addPacket(new MoneyPacket(100, Currency.PLN));
        bag1.addPacket(new MoneyPacket(50, Currency.EUR));

        BagFullOfMoney bag2 = new BagFullOfMoney();
        bag2.addPacket(new MoneyPacket(200, Currency.USD));
        bag2.addPacket(new MoneyPacket(10, Currency.GBP));

        System.out.println("=== Stan początkowy ===");
        System.out.println(ratios);
        System.out.println(bag1);
        System.out.println(bag2);

        // 1. Zmiana kursu jednej waluty (PLN)
        System.out.println("\n=== 1. Zmiana kursu PLN (0.22 -> 0.25) ===");
        ratios.getExchangeRatio(Currency.PLN).setRatio(0.25);
        System.out.println(bag1);
        System.out.println(bag2);

        // 2. Zmiana kursu wspólnej waluty (Odświeżenie wszystkich)
        System.out.println("\n=== 2. Odświeżenie wszystkich kursów ===");
        ratios.refreshAll();
        System.out.println(bag1);
        System.out.println(bag2);

        // 3. Zmiana wyboru wspólnej waluty (na PLN)
        System.out.println("\n=== 3. Zmiana wspólnej waluty na PLN ===");
        Map<Currency, Double> newRatios = new HashMap<>();
        newRatios.put(Currency.EUR, 4.30);
        newRatios.put(Currency.USD, 4.00);
        newRatios.put(Currency.GBP, 5.10);
        newRatios.put(Currency.CHF, 4.50);
        newRatios.put(Currency.PLN, 1.0);
        
        ratios.setBaseCurrency(Currency.PLN, newRatios);
        System.out.println(ratios);
        System.out.println(bag1);
        System.out.println(bag2);
    }
}
