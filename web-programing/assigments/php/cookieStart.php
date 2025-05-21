<?php
setcookie("hello", "Hello world", time() + 60);
echo $_COOKIE['hello'];
