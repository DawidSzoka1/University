package lab9;

import java.util.List;
import java.util.Scanner;

public class Corridor extends Area {
    private final List<Area> doors;

    public Corridor(List<Area> doors) {
        super("Korytarz");
        this.doors = doors;
    }

    @Override
    public Area enter(Player player, Scanner scanner) {
        printHeader(player);
        System.out.println("Korytarz. Widzisz " + doors.size() + " drzwi.");

        int choice = 0;
        while (choice < 1 || choice > doors.size()) {
            System.out.print("Którymi chcesz podążyć? (1-" + doors.size() + "): ");
            try {
                choice = Integer.parseInt(scanner.nextLine().trim());
            } catch (NumberFormatException e) {
                choice = 0;
            }
        }
        return doors.get(choice - 1);
    }
}
