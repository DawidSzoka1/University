<?php
session_start();
$_SESSION['hello'] = "Hello world";
?>
<form action='2.php' method='post'>
<input type='text' name='value'>
<input type='submit' name='wyslij'>
</form>