package lab7;

public class ChocolateProducer {
    
    public Chocolate produceChocolate(String request) {
        Chocolate chocolate = null;
        String lowerRequest = request.toLowerCase();

        if (lowerRequest.contains("mleczna")) {
            MilkChocolate milk = new MilkChocolate();
            if (lowerRequest.contains("mocno")) {
                milk.setExtraMilk(true);
            }
            chocolate = milk;
        } else if (lowerRequest.contains("gorzka")) {
            DarkChocolate dark = new DarkChocolate();
            if (lowerRequest.contains("ekstra")) {
                dark.setExtraDark(true);
            }
            chocolate = dark;
        } else if (lowerRequest.contains("orzechami") || lowerRequest.contains("bakaliami")) {
            chocolate = new NutAndFruitChocolate();
        } else if (lowerRequest.contains("chili") || lowerRequest.contains("solą")) {
            chocolate = new ChiliChocolate();
        }

        if (chocolate != null) {
            chocolate.prepare();
        }

        return chocolate;
    }
}
