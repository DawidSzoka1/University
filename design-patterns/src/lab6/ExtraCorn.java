package lab6;

public class ExtraCorn extends PizzaDecorator {
    public ExtraCorn(iPizza pizza) {
        super(pizza);
    }

    @Override
    public String getDescription() {
        return decorateDescription("kukurydzą");
    }

    @Override
    public double getPrice() {
        return pizza.getPrice() + 2.0 * getAreaFactor();
    }
}
