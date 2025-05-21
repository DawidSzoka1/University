<?php
$dbh = new PDO('mysql:host=localhost;dbname=bazazajecia',
    'root', '');
$sql = 'SELECT id, marka, model, kolor FROM samochod';
foreach ($dbh->query($sql) as $row) {
    print $row['id'] . "\t";
    print $row['marka'] . "\t";
    print $row['model'] . "\t";
    print $row['kolor'] . "\n";
}