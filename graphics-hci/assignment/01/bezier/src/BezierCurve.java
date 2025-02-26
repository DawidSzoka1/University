import javax.swing.*;
import java.awt.*;
import java.awt.event.*;

public class BezierCurve extends JPanel{

    private int x1, y1, ctrlX1, ctrlY1, ctrlX2, ctrlY2, x2, y2;

    public BezierCurve(int x1, int y1, int ctrlX1, int ctrlY1, int ctrlX2, int ctrlY2, int x2, int y2) {
        this.x1 = x1;
        this.y1 = y1;
        this.ctrlX1 = ctrlX1;
        this.ctrlY1 = ctrlY1;
        this.ctrlX2 = ctrlX2;
        this.ctrlY2 = ctrlY2;
        this.x2 = x2;
        this.y2 = y2;
    }

    public void paint(Graphics g) {
        super.paint(g);
        Graphics2D g2d = (Graphics2D) g;

        g2d.setColor(Color.BLUE);

        int prevX = x1, prevY = y1;
        for (double t = 0; t <= 1; t += 0.01) {
            double xt = Math.pow(1 - t, 3) * x1 + 3 * Math.pow(1 - t, 2) * t * ctrlX1 +
                    3 * (1 - t) * Math.pow(t, 2) * ctrlX2 + Math.pow(t, 3) * x2;
            double yt = Math.pow(1 - t, 3) * y1 + 3 * Math.pow(1 - t, 2) * t * ctrlY1 +
                    3 * (1 - t) * Math.pow(t, 2) * ctrlY2 + Math.pow(t, 3) * y2;

            g2d.drawLine(prevX, prevY, (int) xt, (int) yt);
            prevX = (int) xt;
            prevY = (int) yt;
        }

        g2d.setColor(Color.RED);
        g2d.fillOval(x1 - 5, y1 - 5, 10, 10);
        g2d.fillOval(x2 - 5, y2 - 5, 10, 10);

        g2d.setColor(Color.BLACK);
        g2d.fillOval(ctrlX1 - 5, ctrlY1 - 5, 10, 10);
        g2d.fillOval(ctrlX2 - 5, ctrlY2 - 5, 10, 10);
    }


    public static void main(String[] args) {
        int x1 = Integer.parseInt(JOptionPane.showInputDialog("Enter x1:"));
        int y1 = Integer.parseInt(JOptionPane.showInputDialog("Enter y1:"));
        int ctrlX1 = Integer.parseInt(JOptionPane.showInputDialog("Enter ctrlX1:"));
        int ctrlY1 = Integer.parseInt(JOptionPane.showInputDialog("Enter ctrlY1:"));
        int ctrlX2 = Integer.parseInt(JOptionPane.showInputDialog("Enter ctrlX2:"));
        int ctrlY2 = Integer.parseInt(JOptionPane.showInputDialog("Enter ctrlY2:"));
        int x2 = Integer.parseInt(JOptionPane.showInputDialog("Enter x2:"));
        int y2 = Integer.parseInt(JOptionPane.showInputDialog("Enter y2:"));

        JFrame frame = new JFrame("Bezier Curve");
        frame.setSize(400, 300);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(new BezierCurve(x1, y1, ctrlX1, ctrlY1, ctrlX2, ctrlY2, x2, y2));
        frame.setVisible(true);
    }
}
