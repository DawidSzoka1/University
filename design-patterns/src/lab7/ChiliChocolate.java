package lab7;

public class ChiliChocolate extends Chocolate {
    public ChiliChocolate() {
        super("z chili i solą morską");
    }

    @Override
    public void prepare() {
        addIngredient("kakao");
        addIngredient("cukier trzcinowy");
        addIngredient("papryczka chili");
        addIngredient("sól morska");
    }
}
