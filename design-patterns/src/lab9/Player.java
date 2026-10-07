package lab9;

public class Player {
    private static int score = 0;
    private static int lives = 3;
    private static boolean won = false;
    private static boolean gameOver = false;

    public static void addScore(int points) {
        score += points;
    }

    public static void loseLife() {
        lives--;
        if (lives <= 0) {
            gameOver = true;
        }
    }

    public static void win() {
        won = true;
        gameOver = true;
    }

    public static void lose() {
        gameOver = true;
    }

    public static int getScore() { return score; }
    public static int getLives() { return lives; }
    public static boolean isWon() { return won; }
    public static boolean isGameOver() { return gameOver; }

    public static String getStatus() {
        StringBuilder sb = new StringBuilder();
        sb.append("Wynik: ").append(score).append("  Życia: ");
        for (int i = 0; i < lives; i++) sb.append("♥");
        for (int i = lives; i < 3; i++) sb.append("♡");
        return sb.toString();
    }
}
