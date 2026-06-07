package lab10;

/**
 * Konkretny Implementor: telewizor.
 * Cecha unikalna: obsługa obrazu (pictureEnabled, format obrazu).
 */
public class Tv extends BaseDevice {
    private boolean pictureEnabled = true;
    private String pictureFormat = "16:9";

    @Override
    protected String name() {
        return "Telewizor";
    }

    /** Cecha unikalna dla TV - włączanie/wyłączanie obrazu. */
    public void togglePicture() {
        pictureEnabled = !pictureEnabled;
        System.out.println(name() + ": obraz " + (pictureEnabled ? "włączony" : "wyłączony") + ".");
    }

    public void setPictureFormat(String format) {
        this.pictureFormat = format;
        System.out.println(name() + ": zmieniono format obrazu na " + format + ".");
    }

    public boolean isPictureEnabled() {
        return pictureEnabled;
    }

    public String getPictureFormat() {
        return pictureFormat;
    }
}
