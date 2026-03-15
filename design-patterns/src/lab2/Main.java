package lab2;

public class Main {
    public static void main(String[] args) {
        Shop mdk = new Shop("MDK");
        Shop haleRakowieckie = new Shop("Hale Rakowieckie");
        Shop spolem = new Shop("Społem");
        Shop pewex = new Shop("Pewex");

        LadyQueue paniMaria = new LadyQueue("Maria", "Piórecka");
        LadyQueue paniHalina = new LadyQueue("Halina", "Kowalska");
        LadyQueue paniKrystyna = new LadyQueue("Krystyna", "Nowak");
        LadyQueue paniJadwiga = new LadyQueue("Jadwiga", "Wiśniewska");
        LadyQueue paniBozena = new LadyQueue("Bożena", "Wójcik");


        mdk.addObserver(paniMaria);
        mdk.addObserver(paniHalina);
        mdk.addObserver(paniKrystyna);

        spolem.addObserver(paniMaria);

        pewex.addObserver(paniJadwiga);


        System.out.println("--- Dzień 1 ---");
        mdk.throwStuff("mięsa");

        System.out.println("\n--- Dzień 2 ---");
        spolem.throwStuff("ekspresów do kawy");

        System.out.println("\n--- Dzień 3 ---");
        haleRakowieckie.throwStuff("cukru");

        System.out.println("\n--- Dzień 4 ---");
        pewex.throwStuff("sól");

        pewex.addObserver(paniBozena);
        haleRakowieckie.addObserver(paniBozena);
        mdk.addObserver(paniBozena);
        spolem.addObserver(paniBozena);

        System.out.println("\n--- Dzień 5 ---");
        mdk.throwStuff("mięsa");
        spolem.throwStuff("ekspresów do kawy");
        haleRakowieckie.throwStuff("cukru");
        pewex.throwStuff("sól");
    }
}
