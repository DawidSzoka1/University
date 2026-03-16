package lab3;

import java.util.Random;
import java.util.Scanner;

public class Guessing extends Game{

    Scanner scanner;
    Random random;
    private int result;
    private int count;
    private boolean isWinner = false;
    private int guess;

    @Override
    void initialize() {
        System.out.println("Zgadnij liczbę od 1 do 30. Masz 4 próby.");
        scanner = new Scanner(System.in);
        random = new Random();
        result = random.nextInt(30) + 1;
        isWinner = false;
        count = 0;
    }

    @Override
    boolean gameOver() {
        return isWinner || count >= 4;
    }

    @Override
    void makeMove() {
        System.out.print("Podaj swój strzał: ");
        guess = scanner.nextInt();
        count++;
    }

    @Override
    void printScreen() {
        if (guess == result) {
            isWinner = true;
        } else if (guess < result) {
            System.out.println("Szukana liczba jest większa.");
        } else {
            System.out.println("Szukana liczba jest mniejsza.");
        }
    }

    @Override
    void onEnd() {
        System.out.println("-----------------");
        if (isWinner) {
            System.out.println("Gratulacje! Wygrałeś w " + count + " próbach.");
        } else {
            System.out.println("Przegrałeś. Szukana liczba to: " + result);
        }

        if (scanner != null) {
            scanner.close();
        }
    }
}
