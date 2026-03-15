package lab2;

import java.util.Observable;
import java.util.Observer;

public class LadyQueue implements Observer {
    private String firstName;
    private String lastName;

    LadyQueue(String firstName, String lastName) {
        this.firstName = firstName;
        this.lastName = lastName;
    }

    void buyAll(Shop shop) {
        System.out.println("Pani " + this.toString() + " wykupuje " + shop.toString() + ". Siaty pełne.");
    }

    void buyAll(Shop shop, String item) {
        System.out.println("Pani " + this + " wykupuje " + shop.toString() + ". Siaty pełne " + item + ".");
    }

    @Override
    public String toString() {
        return firstName + " " + lastName;
    }

    @Override
    public void update(Observable o, Object arg) {
        if (o instanceof Shop shop && arg instanceof String item) {
            this.buyAll(shop, item);
        }
        else if (o instanceof Shop shop){
            this.buyAll(shop);
        }
    }
}
