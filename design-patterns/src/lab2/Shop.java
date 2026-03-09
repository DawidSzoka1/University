package lab2;

public class Shop {
    private String name;

    Shop(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }

    void throwStuff(String item) {
        System.out.println("Rzucono dostawę " +  item +  " do " + this.name + "! Tłumy szaleją.");
    }

    @Override
    public String toString() {
        return this.name;
    }
}
