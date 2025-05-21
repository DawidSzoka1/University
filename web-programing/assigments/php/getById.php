<?php
$dbh = new PDO('mysql:host=localhost;dbname=bazazajecia',
    'root', '');
$id = $_GET['id'];

$stm = $dbh->prepare("select * from samochod where id=:id");
$stm->bindParam(':id', $id);
$stm->execute();

$row = $stm->fetch();

if ($row) {
    echo "ID: " . $row['id'] . "<br>";
    echo "Marka: " . $row['marka'] . "<br>";
    echo "Model: " . $row['model'] . "<br>";
    echo "Kolor: " . $row['kolor'] . "<br>";
} else {
    echo "Samochod o podanym ID nie istnieje.";
}
