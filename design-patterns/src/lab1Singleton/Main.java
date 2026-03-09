package lab1Singleton;

public class Main {
    public static void main(String[] args) {
        Shop sk = Shop.getInstance();
        sk.name = "Skelp";
        System.out.println(sk);

        sk.buy();

        sk.close();
        sk.buy();

        sk.open();
        sk.buy();

        sk.renovation();
        sk.buy();

    }
}