import javax.swing.*;
import java.awt.*;

public class Main {
    public static void main(String[] args) {
        JFrame frame = new JFrame("Initials using bezier curve");
        AddNewCurvePanel addNewCurvePanel = new AddNewCurvePanel();
        BezierCurvesPanel panel = new BezierCurvesPanel();
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(600, 600);
        frame.setLayout(new BorderLayout());
        frame.add(panel, BorderLayout.CENTER);
        frame.add(addNewCurvePanel, BorderLayout.PAGE_END);
        frame.setVisible(true);
    }
}