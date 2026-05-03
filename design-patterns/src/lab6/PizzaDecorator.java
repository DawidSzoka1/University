package lab6;

import java.util.Locale;

public abstract class PizzaDecorator implements iPizza {
    protected final iPizza pizza;

    public PizzaDecorator(iPizza pizza) {
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

    protected String decorateDescription(String ingredient) {
        String desc = pizza.getDescription();
        if (desc.contains(" z ")) {
            return desc + ", " + ingredient;
        } else {
            return desc + " z " + ingredient;
        }
    }

    @Override
    public String toString() {
        return String.format(Locale.US, "%s, o cenie %.2fzł.", getDescription(), getPrice());
    }
}
