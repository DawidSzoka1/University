package lab6;

public class Hawajska extends PredefinedPizza {
    public Hawajska(int diameter) {
        super(new ExtraPineapple(new ExtraHam(new ExtraCheese(new Pizza(diameter)))));
    }
}
