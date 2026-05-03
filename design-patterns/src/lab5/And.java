package lab5;

import java.util.ArrayList;
import java.util.List;

public class And extends Expression {
    private List<Expression> children = new ArrayList<>();

    public And(Expression... expressions) {
        for (Expression expr : expressions) {
            children.add(expr);
        }
    }

    public void add(Expression expr) {
        children.add(expr);
    }

    @Override
    public boolean evaluate() {
        if (children.isEmpty()) return false;

        // AND operation: false if at least one child evaluates to false
        for (Expression expr : children) {
            if (!expr.evaluate()) {
                return false;
            }
        }
        return true;
    }
}
