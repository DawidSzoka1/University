package lab6;

public class Pepperoni implements iPizza {
    private final iPizza decoratedPizza;

    public Pepperoni(int diameter) {
        this.decoratedPizza = new ExtraSalami(new ExtraCheese(new Pizza(diameter)));
    }

    @Override
    public String getDescription() {
        return decoratedPizza.getDescription();
    }

    @Override
    public double getPrice() {
        return decoratedPizza.getPrice();
    }

    @Override
    public int getDiameter() {
        return decoratedPizza.getDiameter();
    }

    @Override
    public double getAreaFactor() {
        return decoratedPizza.getAreaFactor();
    }

    @Override
    public String toString() {
        return decoratedPizza.toString();
    }
}
