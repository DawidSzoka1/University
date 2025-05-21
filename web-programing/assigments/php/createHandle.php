<?php
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $marka = $_POST['marka'];
    $model = $_POST['model'];
    $kolor = $_POST['kolor'];
    $dbh = new PDO('mysql:host=localhost;dbname=bazazajecia',
        'root', '');
    $stm = $dbh->prepare(
        "insert into samochod(marka, model, kolor) values (:marka, :model, :kolor)");
    $stm->bindParam(':marka', $marka);
    $stm->bindParam(':model', $model);
    $stm->bindParam(':kolor', $kolor);
    $stm->execute();
    echo "<h2>Samochod stworzyony pomyslnie z formularza:</h2>";
    echo "Marka: " . $marka . "<br>";
    echo "Model: " . $model . "<br>";
    echo "Kolor: " . $kolor;
} else {
    echo "Formularz nie został wysłany metodą POST.";
}
