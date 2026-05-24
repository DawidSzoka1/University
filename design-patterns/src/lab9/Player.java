package lab9;

public class Player {
    private int score = 0;
    private int lives = 3;
    private boolean won = false;
    private boolean gameOver = false;

    public void addScore(int points) {
        score += points;
    }

    public void loseLife() {
        lives--;
        if (lives <= 0) {
            gameOver = true;
        }
    }

    public void win() {
        won = true;
        gameOver = true;
    }

    public void lose() {
        gameOver = true;
    }

    public int getScore() { return score; }
    public int getLives() { return lives; }
    public boolean isWon() { return won; }
    public boolean isGameOver() { return gameOver; }

    public String getStatus() {
        StringBuilder sb = new StringBuilder();
        sb.append("Wynik: ").append(score).append("  Życia: ");
        for (int i = 0; i < lives; i++) sb.append("♥");
        for (int i = lives; i < 3; i++) sb.append("♡");
        return sb.toString();
    }
}
