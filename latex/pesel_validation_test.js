function onlyDigits(str)
{
    for (var i = 0; i < str.length; i++) {
        var c = str.charCodeAt(i);
        if (c < 48 || c > 57) return false;
    }
    return true;
}
function decodeMonth(mm, yy)
{
    var year, month;
    if (mm >= 1 && mm <= 12) { year = 1900 + yy; month = mm; }
    else if (mm >= 21 && mm <= 32) { year = 2000 + yy; month = mm - 20; }
    else if (mm >= 41 && mm <= 52) { year = 2100 + yy; month = mm - 40; }
    else if (mm >= 61 && mm <= 72) { year = 2200 + yy; month = mm - 60; }
    else if (mm >= 81 && mm <= 92) { year = 1800 + yy; month = mm - 80; }
    else return null;
    return [year, month];
}

function validateControlNumber(pesel)
{
    var weights = [1, 3, 7, 9, 1, 3, 7, 9, 1, 3];
    var sum = 0;
    for (var i = 0; i < weights.length; i++) sum += weights[i] * parseInt(pesel[i], 10);
    var remainder = sum - 10 * Math.floor(sum / 10);
    var control = 10 - remainder;
    if (control === 10) control = 0;
    return control === parseInt(pesel[10], 10);
}
function validate()
{
    var pesel = this.getField("pesel").value;
    if (typeof pesel !== "string") {
        pesel = String(pesel);
        if(pesel.length === 10 || pesel.length === 9){
            while (pesel.length < 11) {
                pesel = "0" + pesel;
            }
        }
    }
    if (pesel.length !== 11)
        return { valid: false, reason: "PESEL musi mieć dokładnie 11 cyfr." };
    if (!onlyDigits(pesel))
        return { valid: false, reason: "PESEL powinien zawierać tylko cyfry." };
    if (!validateControlNumber(pesel))
        return { valid: false, reason: "Niepoprawna cyfra kontrolna." };
    var yy = parseInt(pesel.slice(0, 2), 10);
    var mm = parseInt(pesel.slice(2, 4), 10);
    var dd = parseInt(pesel.slice(4, 6), 10);
    var decoded = decodeMonth(mm, yy);
    if (!decoded)
        return { valid: false, reason: "Niepoprawny miesiąc w PESEL." };
    var year = decoded[0];
    var month = decoded[1];
    var date = new Date(year, month - 1, dd);
    if (date.getFullYear() !== year || date.getMonth() + 1 !== month || date.getDate() !== dd)
        return { valid: false, reason: "Niepoprawna data urodzenia w PESEL." };
    var gender = (parseInt(pesel[9], 10) % 2 === 1) ? "Mężczyzna" : "Kobieta";
    return { valid: true, birthDate: { year: year, month: month, day: dd }, gender: gender };
}

function checkPesel()
{
    var res = validate();
    if (!res.valid) {
        this.getField("error").value = res.reason;
        this.getField("date").value = "";
        this.getField("gender").value = "";
    } else {
        this.getField("error").value = "PESEL poprawny";
        var y = res.birthDate.year.toString();
        var m = res.birthDate.month.toString();
        if (m.length < 2) m = '0' + m;
        var d = res.birthDate.day.toString();
        if (d.length < 2) d = '0' + d;
        this.getField("date").value = y + "-" + m + "-" + d;
        this.getField("gender").value = res.gender;
    }
}
function clearFields()
{
    this.getField("pesel").value = "";
    this.getField("error").value = "";
    this.getField("date").value = "";
    this.getField("gender").value = "";
}