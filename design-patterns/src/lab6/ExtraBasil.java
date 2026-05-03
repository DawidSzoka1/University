package lab6;

public class ExtraBasil extends PizzaDecorator {
    public ExtraBasil(iPizza pizza) {
        super(pizza);
    }

    @Override
    public String getDescription() {
        return decorateDescription("bazylią");
    }

    @Override
    public double getPrice() {
        return pizza.getPrice() + 2.0 * getAreaFactor();
    }
}
