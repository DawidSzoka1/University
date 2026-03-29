package lab4;

import java.util.ArrayList;

public class Storage {
    private ArrayList<Memento> changes = new ArrayList<>();
    private int currentChange = -1;

    void save(Memento mm) {
        while (changes.size() > currentChange + 1) {
            changes.removeLast();
        }
        changes.add(mm);
        currentChange++;
    }

    Memento readAndBack() {
        if (currentChange >= 0) {
            Memento mem = changes.get(currentChange);
            currentChange--;
            return mem;
        }
        return null;
    }

    Memento readForward() {
        if (currentChange < changes.size() - 1) {
            currentChange++;
            return changes.get(currentChange);
        }
        return null;
    }
}
