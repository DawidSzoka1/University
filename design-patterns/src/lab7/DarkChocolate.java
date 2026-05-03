package lab7;

public class DarkChocolate extends Chocolate {
    private boolean extraDark = false;

    public DarkChocolate() {
        super("gorzka");
    }

    public void setExtraDark(boolean extraDark) {
        this.extraDark = extraDark;
        if (extraDark) {
            this.type = "ekstra gorzka";
            this.about = "czekolada " + type + ": ";
        }
    }

    @Override
    public void prepare() {
        addIngredient(extraDark ? "dużo kakao" : "kakao");
        addIngredient("odrobina cukru");
    }
}
