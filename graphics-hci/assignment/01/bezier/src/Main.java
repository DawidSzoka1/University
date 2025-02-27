import javax.swing.*;
import java.awt.*;

public class Main {
    public static void initialDS(BezierCurvesPanel panel){
        //D
        panel.bezierCurve.addBezierCurve(new Point(118, 200), new Point(118, 20), new Point(200, 180), new Point(200, 30));
        panel.bezierCurve.addBezierCurve(new Point(118, 200), new Point(118, 20), new Point(118, 200), new Point(118, 20));
        //S
        panel.bezierCurve.addBezierCurve(new Point(223, 201), new Point(239, 120), new Point(265, 243), new Point(280, 123));
        panel.bezierCurve.addBezierCurve(new Point(239, 120), new Point(245, 40), new Point(180, 100), new Point(195, 25));
    }
    public static void main(String[] args) {
        JFrame frame = new JFrame("Initials using bezier curve");
        BezierCurvesPanel panel = new BezierCurvesPanel();
        AddNewCurvePanel addNewCurvePanel = new AddNewCurvePanel(panel);
        initialDS(panel);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(700, 600);
        frame.setLayout(new BorderLayout());
        frame.add(panel, BorderLayout.CENTER);
        frame.add(addNewCurvePanel, BorderLayout.PAGE_START);
        frame.setVisible(true);
    }
}