package lab6;

public class ExtraSalami extends PizzaDecorator {
    public ExtraSalami(iPizza pizza) {
        super(pizza);
    }

    @Override
    public String getDescription() {
        return decorateDescription("salami");
    }

    @Override
    public double getPrice() {
        return pizza.getPrice() + 4.5 * getAreaFactor();
    }
}
