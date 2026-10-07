package lab9;

import java.util.Scanner;

public class Room extends Area {
    private final Area next;
    private final RoomContent content;
    private final int rewardValue;
    public Scanner scanner = new Scanner(System.in);

    public Room(Area next, RoomContent content, int rewardValue) {
        super("Pokój");
        this.next = next;
        this.content = content;
        this.rewardValue = rewardValue;
    }

    @Override
    public Area enter() {
        printHeader();

        if (content == RoomContent.REWARD) {
            System.out.println("W rogu widzisz zapakowany prezent.");
            System.out.print("Chcesz go otworzyć? (T/N): ");
            if (scanner.nextLine().trim().equalsIgnoreCase("T")) {
                Player.addScore(rewardValue);
                System.out.println("Zdobywasz " + rewardValue + " punktów! Wynik: " + Player.getScore());
            }
        } else if (content == RoomContent.BOMB) {
            System.out.println("Widzisz tykający pakunek...");
            System.out.print("Chcesz go dotknąć? (T/N): ");
            if (scanner.nextLine().trim().equalsIgnoreCase("T")) {
                System.out.println("BOOOM!!! Tracisz 1 życie.");
                Player.loseLife();
                if (Player.isGameOver()) {
                    System.out.println("Nie masz już żyć. Koniec wędrówki.");
                    return null;
                }
                System.out.println(Player.getStatus());
            }
        } else {
            System.out.println("Pokój jest pusty.");
        }

        System.out.println("Widzisz jedne drzwi. Wchodzisz...");
        return next;
    }
}
