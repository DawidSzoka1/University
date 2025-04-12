addEventListener("DOMContentLoaded", () => {
    const find = document.getElementById("findGame")
    find.addEventListener("click", (e) => {
        document.getElementById("searching").classList.remove("d-none");
        document.getElementById("lobby-info").classList.add("d-none")
        e.preventDefault()
    })
})