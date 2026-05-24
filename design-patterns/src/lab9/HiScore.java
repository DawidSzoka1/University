package lab9;

import java.io.*;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class HiScore {
    private static final String FILE = "lab9_hiscore.txt";
    private static final int MAX_ENTRIES = 5;

    public static void save(int score) {
        List<Integer> scores = load();
        scores.add(score);
        scores.sort(Comparator.reverseOrder());
        if (scores.size() > MAX_ENTRIES) scores = scores.subList(0, MAX_ENTRIES);

        try (PrintWriter pw = new PrintWriter(new FileWriter(FILE))) {
            for (int s : scores) pw.println(s);
        } catch (IOException e) {
            System.out.println("Błąd zapisu hiscore: " + e.getMessage());
        }
    }

    public static List<Integer> load() {
        List<Integer> scores = new ArrayList<>();
        File f = new File(FILE);
        if (!f.exists()) return scores;

        try (BufferedReader br = new BufferedReader(new FileReader(f))) {
            String line;
            while ((line = br.readLine()) != null) {
                try { scores.add(Integer.parseInt(line.trim())); }
                catch (NumberFormatException ignored) {}
            }
        } catch (IOException e) {
            System.out.println("Błąd odczytu hiscore: " + e.getMessage());
        }
        return scores;
    }

    public static void print() {
        List<Integer> scores = load();
        System.out.println("\n--- HISCORE (TOP " + MAX_ENTRIES + ") ---");
        if (scores.isEmpty()) {
            System.out.println("Brak wyników.");
        } else {
            for (int i = 0; i < scores.size(); i++) {
                System.out.println((i + 1) + ". " + scores.get(i) + " pkt");
            }
        }
    }
}
