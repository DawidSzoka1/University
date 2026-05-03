package lab6;

public class ExtraPepper extends PizzaDecorator {
    public ExtraPepper(iPizza pizza) {
        super(pizza);
    }

    @Override
    public String getDescription() {
        return decorateDescription("papryką");
    }

    @Override
    public double getPrice() {
        return pizza.getPrice() + 2.5 * getAreaFactor();
    }
}
