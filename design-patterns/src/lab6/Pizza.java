package lab6;

import java.util.Locale;

public class Pizza implements iPizza {
    private final int diameter;

    public Pizza(int diameter) {
        this.diameter = diameter;
    }

    @Override
    public String getDescription() {
        return "Pizza o średnicy " + diameter + " cm";
    }

    @Override
    public double getPrice() {
        if (diameter <= 32) {
            return 20.0;
        } else if (diameter <= 50) {
            // Simple linear mapping between 32 and 50 for the base price
            // 32 -> 20, 50 -> 35
            return 20.0 + 15.0 * (diameter - 32) / 18.0;
        } else {
            // Extrapolation for larger sizes
            return 35.0 + 15.0 * (diameter - 50) / 18.0;
        }
    }

    @Override
    public int getDiameter() {
        return diameter;
    }

    @Override
    public double getAreaFactor() {
        return Math.pow(diameter / 32.0, 2);
    }

    @Override
    public String toString() {
        return String.format(Locale.US, "%s, o cenie %.2fzł.", getDescription(), getPrice());
    }
}
