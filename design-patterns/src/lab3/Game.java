package lab3;

public abstract class Game {

    public final void run() {
        initialize();
        while(!gameOver()){
            makeMove();
            printScreen();
        }
        onEnd();
    }
    abstract void  initialize();
    abstract boolean gameOver();
    abstract void makeMove();
    abstract void printScreen();
    abstract void onEnd();

}
