function output() {
    const socket = new WebSocket(`ws://${window.location.host}/ws/trans-server/`);
    const input = document.getElementById("translation").value;
    if (input !== "") {
        const inputDiv = document.createElement("div");
        inputDiv.textContent = `${input}`;
        inputDiv.classList.add("input-item");

        const outputDiv = document.getElementById("output");
        outputDiv.appendChild(inputDiv);
        outputDiv.scrollTop = outputDiv.scrollHeight;

        socket.send(JSON.stringify({ text: input }));
        document.getElementById("translation").innerHTML = "";
    } else {
        alert("Please enter a text");
    }
    socket.onmessage = function (e) {
        const data = JSON.parse(e.data);
        const text = data["translated"];
        const newOutput = document.createElement("div");
        newOutput.textContent = `${text}`;
        newOutput.classList.add("output-item");
        
        const outputDiv = document.getElementById("output");
        outputDiv.appendChild(newOutput);
        outputDiv.scrollTop = outputDiv.scrollHeight;
    };
}

document.getElementById("translation").addEventListener("keydown", function (event) {
    if (event.key === "Enter") {
        output();
    }
});
