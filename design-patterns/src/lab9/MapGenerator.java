package lab9;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class MapGenerator {
    private static final int MAX_DEPTH = 7;
    private static final int TERMINAL_MIN_DEPTH = 3;
    private static final Random rng = new Random();

    public static Area generate() {
        return buildArea(0);
    }

    private static Area buildArea(int depth) {
        if (depth >= MAX_DEPTH) {
            return buildTerminal();
        }

        double roll = rng.nextDouble();

        if (depth >= TERMINAL_MIN_DEPTH) {
            if (roll < 0.08) return new Nirvana();
            if (roll < 0.16) return new BlackHole();
        }

        if (rng.nextBoolean()) {
            return buildRoom(depth);
        } else {
            return buildCorridor(depth);
        }
    }

    private static Room buildRoom(int depth) {
        Area next = buildArea(depth + 1);
        double roll = rng.nextDouble();
        if (roll < 0.35) {
            int reward = 10 + rng.nextInt(91);
            return new Room(next, RoomContent.REWARD, reward);
        } else if (roll < 0.65) {
            return new Room(next, RoomContent.BOMB, 0);
        } else {
            return new Room(next, RoomContent.EMPTY, 0);
        }
    }

    private static Corridor buildCorridor(int depth) {
        int doorCount = 2 + rng.nextInt(4);
        List<Area> doors = new ArrayList<>();
        for (int i = 0; i < doorCount; i++) {
            doors.add(buildArea(depth + 1));
        }
        return new Corridor(doors);
    }

    private static Area buildTerminal() {
        return rng.nextBoolean() ? new Nirvana() : new BlackHole();
    }
}
