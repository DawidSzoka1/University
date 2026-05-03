package lab6;

public class Hawajska implements iPizza {
    private final iPizza decoratedPizza;

    public Hawajska(int diameter) {
        this.decoratedPizza = new ExtraPineapple(new ExtraHam(new ExtraCheese(new Pizza(diameter))));
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
