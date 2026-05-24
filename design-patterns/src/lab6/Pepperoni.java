package lab6;

public class Pepperoni extends PredefinedPizza {
    public Pepperoni(int diameter) {
        super(new ExtraSalami(new ExtraCheese(new Pizza(diameter))));
    }
}
