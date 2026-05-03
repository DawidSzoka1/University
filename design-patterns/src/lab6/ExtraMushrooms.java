package lab6;

public class ExtraMushrooms extends PizzaDecorator {
    public ExtraMushrooms(iPizza pizza) {
        super(pizza);
    }

    @Override
    public String getDescription() {
        return decorateDescription("pieczarkami");
    }

    @Override
    public double getPrice() {
        return pizza.getPrice() + 3.0 * getAreaFactor();
    }
}
