package lab6;

public abstract class PredefinedPizza implements iPizza {
    private final iPizza pizza;

    protected PredefinedPizza(iPizza pizza) {
        this.pizza = pizza;
    }

    @Override
    public String getDescription() {
        return pizza.getDescription();
    }

    @Override
    public double getPrice() {
        return pizza.getPrice();
    }

    @Override
    public int getDiameter() {
        return pizza.getDiameter();
    }

    @Override
    public double getAreaFactor() {
        return pizza.getAreaFactor();
    }

    @Override
    public String toString() {
        return format();
    }
}
