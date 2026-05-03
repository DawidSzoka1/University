package lab6;

public class ExtraCheese extends PizzaDecorator {
    public ExtraCheese(iPizza pizza) {
        super(pizza);
    }

    @Override
    public String getDescription() {
        return decorateDescription("serem");
    }

    @Override
    public double getPrice() {
        return pizza.getPrice() + 3.0 * getAreaFactor();
    }
}
