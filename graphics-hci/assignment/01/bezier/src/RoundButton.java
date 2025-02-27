import javax.swing.*;
import java.awt.*;

class RoundButton extends JButton {
    public RoundButton(String label) {
        super(label);
        setContentAreaFilled(false); // Wyłącza domyślne wypełnienie
        setFocusPainted(false);
        setBorderPainted(false);
    }

    @Override
    protected void paintComponent(Graphics g) {
        Graphics2D g2 = (Graphics2D) g;
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        // Pobieramy kolor tła ustawiony przez setBackground()

        g2.setColor(getBackground());
        if(getModel().isArmed()){
            g.setColor(Color.BLACK);
        }
        g2.fillOval(0, 0, getWidth(), getHeight());

        super.paintComponent(g);
    }

    @Override
    protected void paintBorder(Graphics g) {
        g.setColor(getForeground());
        g.drawOval(0, 0, getWidth() - 1, getHeight() - 1);
    }
}
