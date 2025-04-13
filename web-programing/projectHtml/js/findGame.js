addEventListener("DOMContentLoaded", () => {
    const find = document.getElementById("findGame")
    find.addEventListener("click", (e) => {
        e.preventDefault()
        document.getElementById("searching").classList.remove("d-none");
        document.getElementById("lobby-info").classList.add("d-none")
        setTimeout(() =>{
            window.location.href = "/projectHtml/game.html"
        }, 4000)
    })
})