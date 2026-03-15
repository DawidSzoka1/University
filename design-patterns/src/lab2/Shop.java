package lab2;

import java.util.Observable;

public class Shop extends Observable {
    private String name;

    Shop(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }

    void throwStuff(String item) {
        System.out.println("Rzucono dostawę " +  item +  " do " + this.name + "! Tłumy szaleją.");
        setChanged();
        notifyObservers(item);
    }

    @Override
    public String toString() {
        return this.name;
    }
}
