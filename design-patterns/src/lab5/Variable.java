package lab5;

import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

public class Variable extends Expression {
    private String name;

    private static Map<String, Boolean> environment = new HashMap<>();
    private static Scanner scanner = new Scanner(System.in);

    public Variable(String name) {
        this.name = name;
    }

    public static void clearEnvironment() {
        environment.clear();
    }

    @Override
    public boolean evaluate() {
        if (!environment.containsKey(name)) {
            System.out.print("Podaj wartość dla zmiennej '" + name + "' (1/0 lub true/false): ");
            boolean val = false;
            while (true) {
                String input = scanner.next().trim().toLowerCase();
                if (input.equals("1") || input.equals("true")) {
                    val = true;
                    break;
                } else if (input.equals("0") || input.equals("false")) {
                    val = false;
                    break;
                } else {
                    System.out.print("Błąd. Podaj '1', '0', 'true' lub 'false': ");
                }
            }
            environment.put(name, val);
        }
        return environment.get(name);
    }
}
