import javax.swing.*;
import java.awt.*;

public class Main {
    public static void main(String[] args) {
        JFrame frame = new JFrame("Initials using bezier curve");
        BezierCurvesPanel panel = new BezierCurvesPanel();
        AddNewCurvePanel addNewCurvePanel = new AddNewCurvePanel(panel);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(700, 600);
        frame.setLayout(new BorderLayout());
        frame.add(panel, BorderLayout.CENTER);
        frame.add(addNewCurvePanel, BorderLayout.PAGE_START);
        frame.setVisible(true);
    }
}