import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;

public class AddNewCurvePanel extends JPanel {
    JTextField p0 = new JTextField(5);
    JLabel labelp0 = new JLabel("p0 np 10:20");
    JTextField p1 = new JTextField(5);
    JLabel labelp1 = new JLabel("p1 np 10:20");
    JTextField pk0 = new JTextField(5);
    JLabel labelpk0 = new JLabel("pk0 np 10:20");
    JTextField pk1 = new JTextField(5);
    JLabel labelpk1 = new JLabel("pk1 np 10:20");
    JButton create = new JButton("Add curve");

    public Point convertStringToPoint(String point) {
        try {
            String[] tab = point.split(":");
            if (tab.length != 2) {
                throw new Exception("Must by x:y");
            }
            int x = Integer.parseInt(tab[0]);
            int y = Integer.parseInt(tab[1]);
            return new Point(x, y);
        } catch (Exception e) {
            JOptionPane.showMessageDialog(this, e.getMessage());
            return null;
        }

    }

    AddNewCurvePanel(BezierCurvesPanel panel) {
        setLayout(new FlowLayout());
        add(labelp0);
        add(p0);
        add(labelp1);
        add(p1);
        add(labelpk0);
        add(pk0);
        add(labelpk1);
        add(pk1);
        create.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                panel.bezierCurve.addBezierCurve(
                        convertStringToPoint(p0.getText()),
                        convertStringToPoint(p1.getText()),
                        convertStringToPoint(pk0.getText()),
                        convertStringToPoint(pk1.getText()));
                p0.setText("");
                p1.setText("");
                pk0.setText("");
                pk1.setText("");
            }
        });
        add(create);
    }

}
