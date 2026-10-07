package lab6;

public class Main {
    public static void main(String[] args) {
        iPizza p1 = new ExtraHam(new ExtraMozzarella(new ExtraCheese(new Pizza(50))));
        System.out.println(p1);

        iPizza p2 = new Pepperoni(32);
        System.out.println(p2);

        iPizza p3 = new Hawajska(50);
        System.out.println(p3);

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
        
        iPizza p5 = new ExtraHam(new ExtraHam(new Pizza(32)));
        System.out.println(p5);
    }
}
