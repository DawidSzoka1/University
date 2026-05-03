package lab6;

public class ExtraMozzarella extends PizzaDecorator {
    public ExtraMozzarella(iPizza pizza) {
        super(pizza);
    }

    @Override
    public String getDescription() {
        return decorateDescription("mozarellą");
    }

    @Override
    public double getPrice() {
        return pizza.getPrice() + 4.0 * getAreaFactor();
    }
}
