import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;
import java.util.List;


public class BezierCurvesPanel extends JPanel {
    final List<Point[]> curves = new ArrayList<>();
    final List<JButton[]> controlButtons = new ArrayList<>();
    BezierCurve bezierCurve;

    public BezierCurvesPanel() {
        setLayout(null); // Ustawienie ręcznego rozmieszczania komponentów
        bezierCurve = new BezierCurve(this);
        // Dodajemy kilka krzywych na start
        bezierCurve.addBezierCurve(new Point(118, 200), new Point(118, 20), new Point(200, 180), new Point(200, 30));
        bezierCurve.addBezierCurve(new Point(118, 200), new Point(118, 20), new Point(118, 200), new Point(118, 20));

    }


    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g;

        for (Point[] controlPoints : curves) {
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g2.setStroke(new BasicStroke(2));
            g2.setColor(Color.BLUE);
            bezierCurve.drawBezierCurve(g2, controlPoints);
        }
    }


    public static void main(String[] args) {
        JFrame frame = new JFrame("Multiple Bézier Curves");
        BezierCurvesPanel panel = new BezierCurvesPanel();

        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(600, 600);
        frame.setLayout(new BorderLayout());
        frame.add(panel, BorderLayout.CENTER);
        frame.setVisible(true);
    }
}
