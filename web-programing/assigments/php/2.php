<?php
session_start();
$value = $_POST['value'];
if(isset($_SESSION['value'])){
    echo $_SESSION['value'] . "<br>";
}else{
    $_SESSION['value'] = $value;
}

if (isset($_SESSION['hello'])) {
    echo $_SESSION['hello'];
} else {
    echo "Brak danych w sesji.";
}
?>