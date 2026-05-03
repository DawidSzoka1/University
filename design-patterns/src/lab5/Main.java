package lab5;

public class Main {
    public static void main(String[] args) {
        System.out.println("--- TEST 1 ---");
        // Expression: ((false AND true AND (NOT true) AND false) OR true OR false) AND true
        Expression innerAnd = new And(
                new Constant(false),
                new Constant(true),
                new Not(new Constant(true)),
                new Constant(false)
        );
        Expression innerOr = new Or(
                innerAnd,
                new Constant(true),
                new Constant(false)
        );
        Expression test1 = new And(innerOr, new Constant(true));

        System.out.println("Wynik Testu 1: " + test1.evaluate()); // Expected: true


        System.out.println("\n--- TEST 2 ---");
        Variable.clearEnvironment(); // Resetting variables
        // Expression: (x OR (NOT true) OR false)
        Expression test2 = new Or(
                new Variable("x"),
                new Not(new Constant(true)),
                new Constant(false)
        );
        System.out.println("Wynik Testu 2: " + test2.evaluate());


        System.out.println("\n--- TEST 3 ---");
        Variable.clearEnvironment();
        // Expression: (x AND NOT (y))
        Expression test3 = new And(
                new Variable("x"),
                new Not(new Variable("y"))
        );
        System.out.println("Wynik Testu 3 (x AND !y): " + test3.evaluate());
    }
}
