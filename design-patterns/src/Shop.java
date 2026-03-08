public class Shop {
    private Shop() {
        state = new StateOpen();
    }

    public String name;
    private volatile static Shop instance;
    private ShopState state;

    public static Shop getInstance() {
        if (instance == null) {
            instance = new Shop();
        }
        return instance;
    }


    public void buy() {
        this.state.buy();
    }

    public void close() {
        this.state = new StateClose();
    }

    public void open() {
        this.state = new StateOpen();
    }

    public void renovation() {
        this.state = new StateRenovation();
    }

    @Override
    public String toString() {
        return name;
    }
}
