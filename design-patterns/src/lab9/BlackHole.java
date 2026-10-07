package lab9;

import java.util.Scanner;

public class BlackHole extends Area {
    public BlackHole() {
        super("Czarna Dziura");
    }

    @Override
    public Area enter() {
        printHeader();
        System.out.println("Wpadasz w czarną dziurę i znikasz na zawsze. Przegrywasz!");
        Player.lose();
        return null;
    }
}
