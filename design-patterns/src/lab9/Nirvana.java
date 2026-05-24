package lab9;

import java.util.Scanner;

public class Nirvana extends Area {
    public Nirvana() {
        super("Nirwana");
    }

    @Override
    public Area enter(Player player, Scanner scanner) {
        printHeader(player);
        System.out.println("Czujesz spokój i harmonię. Wygrywasz! ♥♥♥");
        player.win();
        return null;
    }
}
