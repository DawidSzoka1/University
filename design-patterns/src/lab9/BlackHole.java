package lab9;

import java.util.Scanner;

public class BlackHole extends Area {
    public BlackHole() {
        super("Czarna Dziura");
    }

    @Override
    public Area enter(Player player, Scanner scanner) {
        printHeader(player);
        System.out.println("Wpadasz w czarną dziurę i znikasz na zawsze. Przegrywasz!");
        player.lose();
        return null;
    }
}
