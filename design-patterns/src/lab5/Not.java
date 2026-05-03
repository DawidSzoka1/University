package lab5;

public class Not extends Expression {
    private Expression operand;

    public Not(Expression operand) {
        this.operand = operand;
    }

    @Override
    public boolean evaluate() {
        return !operand.evaluate();
    }
}
