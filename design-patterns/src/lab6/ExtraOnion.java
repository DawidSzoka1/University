package lab6;

public class ExtraOnion extends PizzaDecorator {
    public ExtraOnion(iPizza pizza) {
        super(pizza);
    }

    @Override
    public String getDescription() {
        return decorateDescription("cebulą");
    }

    @Override
    public double getPrice() {
        return pizza.getPrice() + 1.5 * getAreaFactor();
    }
}
