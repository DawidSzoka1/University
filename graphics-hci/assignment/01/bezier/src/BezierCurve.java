package src;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;

// Klasa odpowiedzialna za dodawanie i rysowanie pojedynczej krzywej Béziera
public class BezierCurve {
    private BezierCurvesPanel panel;

    // Konstruktor – przyjmuje panel, na którym będą rysowane krzywe
    BezierCurve(BezierCurvesPanel panel) {
        this.panel = panel;
    }

    // Dodaje nową krzywą Béziera do panelu, wraz z punktami kontrolnymi jako przyciski
    public void addBezierCurve(Point start, Point end, Point control1, Point control2) {
        // Tworzymy tablicę punktów kontrolnych (4 punkty: początek, 2 kontrolne, koniec)
        Point[] controlPoints = {start, control1, control2, end};
        panel.curves.add(controlPoints); // Dodajemy krzywą do listy w panelu

        JButton[] buttons = new JButton[4]; // Przyciski do przesuwania punktów kontrolnych

        for (int i = 0; i < 4; i++) {
            // Tworzenie przycisku w miejscu punktu kontrolnego
            JButton button = new RoundButton(""); // Zaokrąglony przycisk (custom class)
            button.setBounds(controlPoints[i].x - 10, controlPoints[i].y - 10, 20, 20);
            button.setBackground(Color.RED); // Domyślnie czerwony kolor

            int index = i; // potrzebne w lambda wewnątrz listenera

            // Dodajemy możliwość przeciągania przycisków myszką
            button.addMouseMotionListener(new MouseMotionAdapter() {
                @Override
                public void mouseDragged(MouseEvent e) {
                    // Przeliczanie pozycji względem panelu
                    Point newPoint = SwingUtilities.convertPoint(button, e.getPoint(), panel);

                    // Aktualizacja punktu kontrolnego i przycisku
                    controlPoints[index].setLocation(newPoint.getLocation());
                    button.setBounds(newPoint.x - 10, newPoint.y - 10, button.getWidth(), button.getHeight());

                    panel.repaint(); // Odświeżenie widoku
                }
            });

            // Ustawienie koloru w zależności od typu punktu (początek/koniec vs kontrolny)
            if (i == 0 || i == 3) {
                button.setBackground(Color.RED); // Punkt początkowy i końcowy
            } else {
                button.setBackground(Color.LIGHT_GRAY); // Punkty kontrolne
            }

            // Dodanie przycisku do panelu i tablicy
            panel.add(button);
            buttons[i] = button;
        }

        // Dodanie zestawu przycisków do listy w panelu
        panel.controlButtons.add(buttons);
        panel.repaint(); // Odświeżenie panelu
    }

    // Rysuje gładką krzywą Béziera na podstawie 4 punktów kontrolnych
    public void drawBezierCurve(Graphics2D g2, Point[] points) {
        int steps = 1000; // Im więcej kroków, tym gładsza krzywa

        // Łączenie kolejnych punktów linią, tworząc krzywą Béziera
        for (int i = 0; i < steps; i++) {
            double t1 = (double) i / steps;
            double t2 = (double) (i + 1) / steps;

            Point p1 = bezierPoint(points, t1);
            Point p2 = bezierPoint(points, t2);

            g2.drawLine(p1.x, p1.y, p2.x, p2.y); // Rysowanie odcinka między p1 a p2
        }

        // Rysujemy linie pomocnicze (przerywane) między punktami kontrolnymi
        g2.setColor(Color.GRAY);
        g2.setStroke(new BasicStroke(1, BasicStroke.CAP_BUTT, BasicStroke.JOIN_BEVEL, 0, new float[]{5}, 0));
        g2.drawLine(points[0].x, points[0].y, points[1].x, points[1].y); // linia pomocnicza
        g2.drawLine(points[2].x, points[2].y, points[3].x, points[3].y);
    }

    // Oblicza pojedynczy punkt na krzywej Béziera (4-punktowej) dla danego parametru t
    private Point bezierPoint(Point[] controlPoints, double t) {
        // Wzory Béziera 3. stopnia (cubic)
        double x = Math.pow(1 - t, 3) * controlPoints[0].x +
                3 * Math.pow(1 - t, 2) * t * controlPoints[1].x +
                3 * (1 - t) * Math.pow(t, 2) * controlPoints[2].x +
                Math.pow(t, 3) * controlPoints[3].x;

        double y = Math.pow(1 - t, 3) * controlPoints[0].y +
                3 * Math.pow(1 - t, 2) * t * controlPoints[1].y +
                3 * (1 - t) * Math.pow(t, 2) * controlPoints[2].y +
                Math.pow(t, 3) * controlPoints[3].y;

        return new Point((int) x, (int) y);
    }
}
