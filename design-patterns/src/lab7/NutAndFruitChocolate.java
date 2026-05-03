package lab7;

public class NutAndFruitChocolate extends Chocolate {
    public NutAndFruitChocolate() {
        super("z orzechami i bakaliami");
    }

    @Override
    public void prepare() {
        addIngredient("kakao");
        addIngredient("cukier");
        addIngredient("mleko");
        addIngredient("orzechy laskowe");
        addIngredient("bakalie");
    }
}
