package lab6;

public class Main {
    public static void main(String[] args) {
        // Przykład 1: Pizza z trzema dodatkami (rozmiar 50cm)
        iPizza p1 = new ExtraHam(new ExtraMozzarella(new ExtraCheese(new Pizza(50))));
        System.out.println(p1);

        // Przykład 2: Predefiniowana pizza Pepperoni (rozmiar 32cm)
        iPizza p2 = new Pepperoni(32);
        System.out.println(p2);

        // Przykład 3: Predefiniowana pizza Hawajska (rozmiar 50cm)
        iPizza p3 = new Hawajska(50);
        System.out.println(p3);

        // Przykład 4: Customowa pizza z wieloma dodatkami i nietypowym rozmiarem
        iPizza p4 = new ExtraBasil(
                        new ExtraPepper(
                            new ExtraOnion(
                                new ExtraMushrooms(
                                    new ExtraCheese(new Pizza(42))
                                )
                            )
                        )
                    );
        System.out.println(p4);
        
        // Przykład 5: Pizza z podwójnym składnikiem
        iPizza p5 = new ExtraHam(new ExtraHam(new Pizza(32)));
        System.out.println(p5);
    }
}
