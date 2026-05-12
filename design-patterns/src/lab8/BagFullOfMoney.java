package lab8;

import java.util.ArrayList;
import java.util.List;

public class BagFullOfMoney {
    private final List<MoneyPacket> packets;

    public BagFullOfMoney() {
        this.packets = new ArrayList<>();
    }

    public void addPacket(MoneyPacket packet) {
        packets.add(packet);
    }

    public double getValue() {
        double total = 0;
        for (MoneyPacket p : packets) {
            total += p.getValue();
        }
        return total;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Torba z pieniędzmi (Suma: ").append(String.format("%.2f", getValue()))
          .append(" ").append(CurrencyRatios.getInstance().getBaseCurrency()).append("):\n");
        for (MoneyPacket p : packets) {
            sb.append("  - ").append(p).append("\n");
        }
        return sb.toString();
    }
}
