package lab7;

public class Main {
    public static void main(String[] args) {
        ChocolateProducer producer = new ChocolateProducer();

        // Standardowe czekolady
        printProduction(producer, "mleczna");
        printProduction(producer, "gorzka");
        printProduction(producer, "czekolada z orzechami i bakaliami");

        // Warianty
        printProduction(producer, "mocno mleczna");
        printProduction(producer, "ekstra gorzka");

        // Własna receptura
        printProduction(producer, "czekolada z chili i solą morską");
        
        // Nieznana
        printProduction(producer, "biała");
    }

    private static void printProduction(ChocolateProducer producer, String type) {
        System.out.println("Zamówienie: " + type);
        Chocolate chocolate = producer.produceChocolate(type);
        if (chocolate != null) {
            System.out.println("Produkcja zakończona: " + chocolate.giveChocolate());
        } else {
            System.out.println("Błąd: Nie mamy takiej czekolady w ofercie.");
        }
        System.out.println("--------------------------------------------------");
    }
}
