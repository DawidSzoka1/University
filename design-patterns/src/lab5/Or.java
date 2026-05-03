package lab5;

import java.util.ArrayList;
import java.util.List;

public class Or extends Expression {
    private List<Expression> children = new ArrayList<>();

    public Or(Expression... expressions) {
        for (Expression expr : expressions) {
            children.add(expr);
        }
    }

    public void add(Expression expr) {
        children.add(expr);
    }

    @Override
    public boolean evaluate() {
        // OR operation: true if at least one child evaluates to true
        for (Expression expr : children) {
            if (expr.evaluate()) {
                return true;
            }
        }
        return false;
    }
}