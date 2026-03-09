package lab2;

public class LadyQueue {
    private String firstName;
    private String lastName;

    LadyQueue(String firstName, String lastName) {
        this.firstName = firstName;
        this.lastName = lastName;
    }

    void buyAll(Shop shop) {
        System.out.println("Pani " + this.toString() + " wykupuje " + shop.toString() + ". Siaty pełne.");
    }

    @Override
    public String toString() {
        return firstName + " " + lastName;
    }
}
