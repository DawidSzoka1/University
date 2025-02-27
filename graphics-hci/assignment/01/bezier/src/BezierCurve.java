import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;


public class BezierCurve {
    private BezierCurvesPanel panel;

    BezierCurve(BezierCurvesPanel panel) {
        this.panel = panel;
    }

    public void addBezierCurve(Point start, Point end, Point control1, Point control2) {
        Point[] controlPoints = {start, control1, control2, end};
        panel.curves.add(controlPoints);
        JButton[] buttons = new JButton[4];

        for (int i = 0; i < 4; i++) {
            JButton button = new RoundButton("");
            button.setBounds(controlPoints[i].x - 10, controlPoints[i].y - 10, 20, 20);
            button.setBackground(Color.RED);
            int index = i;
            button.addMouseMotionListener(new MouseMotionAdapter() {
                @Override
                public void mouseDragged(MouseEvent e) {
                    Point newPoint = SwingUtilities.convertPoint(button, e.getPoint(), panel);
                    controlPoints[index].setLocation(newPoint.getLocation());
                    button.setBounds(newPoint.x - 10, newPoint.y - 10, button.getWidth(), button.getHeight());
                    panel.repaint();
                }
            });
            if (i == 0 || i == 3) {
                button.setBackground(Color.RED); // Punkty startu i końca
            } else {
                button.setBackground(Color.LIGHT_GRAY); // Punkty kontrolne
            }
            panel.add(button);
            buttons[i] = button;

        }

        panel.controlButtons.add(buttons);
        panel.repaint();
    }

    public void drawBezierCurve(Graphics2D g2, Point[] points){
        // rysujemy mala linie miedzy dwoma punktami dzieki t1 i t2
        int steps = 1000;
        for (int i = 0; i < steps; i++) {
            double t1 = (double) i / steps;
            double t2 = (double) (i + 1) / steps;

            Point p1 = bezierPoint(points, t1);
            Point p2 = bezierPoint(points, t2);

            g2.drawLine(p1.x, p1.y, p2.x, p2.y);
        }
        // dodajemy szara linie przerywana do punktow kontrolnych
        g2.setColor(Color.GRAY);
        g2.setStroke(new BasicStroke(1, BasicStroke.CAP_BUTT, BasicStroke.JOIN_BEVEL, 0, new float[]{5}, 0));
        g2.drawLine(points[0].x, points[0].y, points[1].x, points[1].y);
        g2.drawLine(points[2].x, points[2].y, points[3].x, points[3].y);
    }

    private Point bezierPoint(Point[] controlPoints, double t) {
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
