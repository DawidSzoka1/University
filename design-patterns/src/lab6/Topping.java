package lab6;

public abstract class Topping extends PizzaDecorator {
    private final String ingredient;
    private final double basePrice;

    protected Topping(iPizza pizza, String ingredient, double basePrice) {
        super(pizza);
        this.ingredient = ingredient;
        this.basePrice = basePrice;
    }

    @Override
    public String getDescription() {
        return decorateDescription(ingredient);
    }

    @Override
    public double getPrice() {
        return pizza.getPrice() + basePrice * getAreaFactor();
    }
}
