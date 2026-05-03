package lab7;

public class MilkChocolate extends Chocolate {
    private boolean extraMilk = false;

    public MilkChocolate() {
        super("mleczna");
    }

    public void setExtraMilk(boolean extraMilk) {
        this.extraMilk = extraMilk;
        if (extraMilk) {
            this.type = "mocno mleczna";
            this.about = "czekolada " + type + ": ";
        }
    }

    @Override
    public void prepare() {
        addIngredient("kakao");
        addIngredient("cukier");
        addIngredient(extraMilk ? "dużo mleka" : "mleko");
    }
}
