import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.geom.Path2D;
import java.util.ArrayList;
import java.util.List;
//TODO control points different color and moving points
public class BezierInitials extends JPanel {
    private final List<Point> controlPoints;

    public BezierInitials() {
        this.controlPoints = new ArrayList<>();
        controlPoints.add(new Point(50, 150));
        controlPoints.add(new Point(100, 50));
        controlPoints.add(new Point(200, 50));
        controlPoints.add(new Point(250, 150));

        addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                for (Point p : controlPoints) {
                    if (p.distance(e.getPoint()) < 10) {
                        System.out.println("Kliknięto punkt: " + p);
                    }
                }
            }
        });
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g;
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g2.setColor(Color.BLACK);
        g2.setStroke(new BasicStroke(3));

        drawBezierCurve(g2);
        drawControlPoints(g2);
    }

    private void drawBezierCurve(Graphics2D g2) {
        Path2D.Double path = new Path2D.Double();
        path.moveTo(controlPoints.get(0).x, controlPoints.get(0).y);
        path.curveTo(
                controlPoints.get(1).x, controlPoints.get(1).y,
                controlPoints.get(2).x, controlPoints.get(2).y,
                controlPoints.get(3).x, controlPoints.get(3).y
        );
        g2.draw(path);
    }

    private void drawControlPoints(Graphics2D g2) {
        g2.setColor(Color.RED);
        for (Point p : controlPoints) {
            g2.fillOval(p.x - 5, p.y - 5, 10, 10);
        }
    }

    public static void main(String[] args) {
        JFrame frame = new JFrame("Krzywa Béziera");
        BezierInitials panel = new BezierInitials();
        frame.add(panel);
        frame.setSize(400, 300);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
    }
}
