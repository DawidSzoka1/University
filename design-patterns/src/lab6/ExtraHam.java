package lab6;

public class ExtraHam extends PizzaDecorator {
    public ExtraHam(iPizza pizza) {
        super(pizza);
    }

    @Override
    public String getDescription() {
        return decorateDescription("szynką");
    }

    @Override
    public double getPrice() {
        return pizza.getPrice() + 5.0 * getAreaFactor();
    }
}
