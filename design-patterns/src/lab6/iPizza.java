package lab6;

public interface iPizza {
    String getDescription();
    double getPrice();
    int getDiameter();
    double getAreaFactor();

    default String format() {
        return String.format(java.util.Locale.forLanguageTag("pl"), "%s, o cenie %.2fzł.", getDescription(), getPrice());
    }
}
