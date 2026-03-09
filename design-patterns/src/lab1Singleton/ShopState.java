package lab1Singleton;

interface ShopState {
    void buy();
}

class StateOpen implements ShopState {
    @Override
    public void buy() {
        System.out.println("Kupowanie zakończone sukcesem - sklep jest otwarty.");
    }
}

class StateClose implements ShopState {
    @Override
    public void buy() {
        System.out.println("Nie można kupić, sklep jest obecnie zamknięty.");
    }
}

class StateRenovation implements ShopState {
    @Override
    public void buy() {
        System.out.println("Nie można kupić, bo trwa remanent.");
    }
}