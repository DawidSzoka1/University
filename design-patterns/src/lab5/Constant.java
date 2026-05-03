package lab5;

public class Constant extends Expression {
    private boolean value;

    public Constant(boolean value) {
        this.value = value;
    }

    @Override
    public boolean evaluate() {
        return value;
    }
}
