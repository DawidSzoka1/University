package lab9;

import java.util.Scanner;

public class Game {
    private final Scanner scanner = new Scanner(System.in);

    public void start() {
        System.out.println("=== GRA PRZYGODOWA ===");
        System.out.println("Cel: dotrzyj do Nirwany i wygraj!");
        HiScore.print();

        String answer;
        do {
            playRound();
            System.out.print("\nZagrać jeszcze raz? (T/N): ");
            answer = scanner.nextLine().trim();
        } while (answer.equalsIgnoreCase("T"));

        System.out.println("Do widzenia!");
        scanner.close();
    }

    private void playRound() {
        Player player = new Player();
        Area current = MapGenerator.generate();
        System.out.println("\n--- Nowa gra. Masz 3 życia. Powodzenia! ---");

        while (current != null && !player.isGameOver()) {
            current = current.enter(player, scanner);
        }

        System.out.println();
        if (player.isWon()) {
            System.out.println("*** WYGRANA! Wynik: " + player.getScore() + " pkt ***");
            HiScore.save(player.getScore());
            HiScore.print();
        } else {
            System.out.println("*** PRZEGRANA. Wynik: " + player.getScore() + " pkt ***");
        }
    }
}
