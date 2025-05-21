<?php
$dbh = new PDO('mysql:host=localhost;dbname=bazazajecia',
    'root', '');
$dbh -> exec(
    "insert into samochod(marka, model, kolor) values('cabro', 'modle', 'kolor')"

)
?>