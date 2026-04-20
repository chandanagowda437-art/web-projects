const cells = document.querySelectorAll(".cell");
const statusText = document.getElementById("status");
const resetBtn = document.getElementById("reset");

let currentPlayer = "x"
let gameActive = true
let gameState = ["","","","","","","","",""]

let winConditions = [
    [0,1,2], [3,4,5], [6,7,8], //rows
    [0,3,6], [1,4,7], [2,5,8], //columns
    [0, 4, 8], [2, 4, 6] //diagonals

];

function handleCellClick(e){
    let index = [...cells].indexOf(e.target)

    if (!gameActive || gameState[index] !== "") {
        return;
    }

    gameState[index] = currentPlayer;
    e.target.textContent = currentPlayer

    if (checkWin()) {
        statusText.textContent = `${currentPlayer} win the game!`
        gameActive = false
    }else if (!gameState.includes("")) {
        statusText.textContent = `game draw!`
        gameActive = false
    }else{
        currentPlayer = currentPlayer === "X" ? "O" : "X";
        statusText.textContent = `${currentPlayer}'s turn!`
    }
}

function checkWin() {
    return winConditions.some(combination => {
        return combination.every(index => gameStatr[index] === currentPlayer)
    });
}

function resetGame() {
    currentPlayer = 'X'
    gameActive = true
    gameState = ["","","","","","","","",""]

    cells.forEach(cell => cell.textContent = "")
    statusText.textContent = `${currentPlayer}'s turn`

}

cells.forEach(cell => cell.addEventListener("click", handleCellClick))
resetBtn.addEventListener("click", resetGame)

reserGame() //initialize