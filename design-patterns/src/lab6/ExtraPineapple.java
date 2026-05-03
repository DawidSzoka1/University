package lab6;

public class ExtraPineapple extends PizzaDecorator {
    public ExtraPineapple(iPizza pizza) {
        super(pizza);
    }

    @Override
    public String getDescription() {
        return decorateDescription("ananasem");
    }

    @Override
    public double getPrice() {
        return pizza.getPrice() + 3.5 * getAreaFactor();
    }
}
