package lab9;

import java.util.Scanner;

public class Room extends Area {
    private final Area next;
    private final RoomContent content;
    private final int rewardValue;

    public Room(Area next, RoomContent content, int rewardValue) {
        super("Pokój");
        this.next = next;
        this.content = content;
        this.rewardValue = rewardValue;
    }

    @Override
    public Area enter(Player player, Scanner scanner) {
        printHeader(player);

        if (content == RoomContent.REWARD) {
            System.out.println("W rogu widzisz zapakowany prezent.");
            System.out.print("Chcesz go otworzyć? (T/N): ");
            if (scanner.nextLine().trim().equalsIgnoreCase("T")) {
                player.addScore(rewardValue);
                System.out.println("Zdobywasz " + rewardValue + " punktów! Wynik: " + player.getScore());
            }
        } else if (content == RoomContent.BOMB) {
            System.out.println("Widzisz tykający pakunek...");
            System.out.print("Chcesz go dotknąć? (T/N): ");
            if (scanner.nextLine().trim().equalsIgnoreCase("T")) {
                System.out.println("BOOOM!!! Tracisz 1 życie.");
                player.loseLife();
                if (player.isGameOver()) {
                    System.out.println("Nie masz już żyć. Koniec wędrówki.");
                    return null;
                }
                System.out.println(player.getStatus());
            }
        } else {
            System.out.println("Pokój jest pusty.");
        }

        System.out.println("Widzisz jedne drzwi. Wchodzisz...");
        return next;
    }
}
