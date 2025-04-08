import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;
import java.util.List;

// Klasa JPanel odpowiedzialna za rysowanie i zarządzanie wieloma krzywymi Béziera
public class BezierCurvesPanel extends JPanel {

    // Lista wszystkich krzywych Béziera – każda krzywa to tablica punktów kontrolnych
    final List<Point[]> curves = new ArrayList<>();

    // Lista przycisków kontrolnych (dla GUI) odpowiadających punktom kontrolnym
    final List<JButton[]> controlButtons = new ArrayList<>();

    // Obiekt klasy pomocniczej do rysowania krzywych
    BezierCurve bezierCurve;

    // Konstruktor panelu
    public BezierCurvesPanel() {
        setLayout(null); // Wyłączenie domyślnego layoutu – pozycjonowanie ręczne
        bezierCurve = new BezierCurve(this); // Inicjalizacja klasy odpowiedzialnej za rysowanie krzywych
    }

    // Nadpisana metoda odpowiedzialna za rysowanie komponentu
    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g); // Wywołanie domyślnego rysowania (czyści panel)
        Graphics2D g2 = (Graphics2D) g; // Rzutowanie do Graphics2D dla lepszej kontroli

        // Rysowanie każdej z krzywych Béziera
        for (Point[] controlPoints : curves) {
            // Włączenie wygładzania linii (antyaliasing)
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            // Ustawienie grubości pędzla
            g2.setStroke(new BasicStroke(2));

            // Ustawienie koloru rysowanej krzywej
            g2.setColor(Color.BLUE);

            // Rysowanie krzywej Béziera przy użyciu punktów kontrolnych
            bezierCurve.drawBezierCurve(g2, controlPoints);
        }
    }
}
