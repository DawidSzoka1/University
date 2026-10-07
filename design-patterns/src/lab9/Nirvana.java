package lab9;

import java.util.Scanner;

public class Nirvana extends Area {
    public Nirvana() {
        super("Nirwana");
    }

    @Override
    public Area enter() {
        printHeader();
        System.out.println("Czujesz spokój i harmonię. Wygrywasz! ♥♥♥");
        Player.win();
        return null;
    }
}
