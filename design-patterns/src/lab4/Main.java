package lab4;

import java.util.Arrays;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Storage storage = new Storage();

        Scanner scanner = new Scanner(System.in);
        System.out.println("=== Inicjalizacja Macierzy ===");
        int rows = getNumber(scanner);
        int cols = getNumber(scanner);
        Matrix matrix = new Matrix(rows, cols, storage);
        System.out.println("\nRozpoczęto program. Dostępne komendy: s <wiersz> <kolumna> <wartość>, undo, redo, exit");
        System.out.println("Pamiętaj, że wiersze i kolumny są indeksowane od 0!");
        scanner.nextLine();

        while (true) {
            System.out.print("> ");
            String input = scanner.nextLine().trim();
            String[] tokens = input.split(" ");
            if (tokens.length == 0 || tokens[0].isEmpty()) continue;

            String command = tokens[0].toLowerCase();
            if (command.equals("exit")) {
                System.out.println("Zakończenie programu.");
                break;
            } else if (command.equals("undo")) {
                matrix.undo();
                System.out.print(matrix);
            } else if (command.equals("redo")) {
                matrix.redo();
                System.out.print(matrix);
            } else if (command.equals("s") && tokens.length == 4) {
                try {
                    int row = Integer.parseInt(tokens[1]);
                    int col = Integer.parseInt(tokens[2]);
                    double val = Double.parseDouble(tokens[3]);
                    matrix.set(row, col, val);
                    System.out.print(matrix);
                } catch (NumberFormatException e) {
                    System.out.println("Błąd: Argumenty wiersza, kolumny i wartości muszą być liczbami.");
                }
            } else {
                System.out.println("Nieznana komenda lub zły format. Użyj: s <wiersz> <kolumna> <wartość>, undo, redo, exit");
            }
        }
        scanner.close();

    }

    static int getNumber(Scanner scanner){
        int number;
        while (true) {
            System.out.print("Podaj liczbę kolumn (liczba całkowita > 0): ");
            if (scanner.hasNextInt()) {
                number = scanner.nextInt();
                if (number > 0) return number;
                System.out.println("Błąd: Liczba kolumn musi być większa od 0!");
            } else {
                System.out.println("Błąd: To nie jest poprawna liczba całkowita!");
                scanner.next();
            }
        }
    }
}
