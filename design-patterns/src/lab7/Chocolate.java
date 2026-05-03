package lab7;

public abstract class Chocolate {
    protected String about;
    protected String type;

    public Chocolate(String type) {
        this.type = type;
        this.about = "czekolada " + type + ": ";
    }

    public abstract void prepare();

    public String giveChocolate() {
        return about;
    }

    protected void addIngredient(String ingredient) {
        if (!about.endsWith(": ")) {
            about += ", ";
        }
        about += ingredient;
    }
}
