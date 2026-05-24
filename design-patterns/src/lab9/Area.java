package lab9;

import java.util.Scanner;

public abstract class Area {
    protected final String name;

    protected Area(String name) {
        this.name = name;
    }

    public abstract Area enter(Player player, Scanner scanner);

    protected void printHeader(Player player) {
        System.out.println("\n--- " + name + " ---");
        System.out.println(player.getStatus());
    }
}
