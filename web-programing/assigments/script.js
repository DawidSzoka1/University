document.addEventListener("DOMContentLoaded", () => {
    const button = document.getElementById("calculate")

    let matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ];
    let fullRow = []
    matrix.forEach((row, rowIndex) => {
        fullRow = []
        row.forEach((element, colIndex) => {
            fullRow.push(element)
        });
        console.log(fullRow)
    });

    button.addEventListener("click", (e) => {
        e.preventDefault()
        const num1 = parseFloat(document.getElementById("num1").value)
        const num2 = parseFloat(document.getElementById("num2").value)
        console.log(num1 + num2)
        document.getElementById("sum").innerText ="Dodawanie: " +  (num1 + num2)
        document.getElementById("diff").innerText = "Odejmowanie: " + (num1 - num2)
        document.getElementById("mul").innerText ="Mnozenie: " +  (num1 * num2)
        if(num2 !== 0){
            document.getElementById("dev").innerText = "dzielenie: " +  (num1 / num2)
        }else{
            document.getElementById("dev").innerText = "Nie mozna przez 0"
        }
    })
})