document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("predictForm").addEventListener("submit", async function (event) {
        event.preventDefault();

        let formData = new FormData(this);
        let response = await fetch("/predict", { method: "POST", body: formData });
        let result = await response.json();

        let resultDiv = document.getElementById("result");
        if (result.error) {
            resultDiv.innerText = "Error: " + result.error;
            resultDiv.style.color = "red";
        } else {
            resultDiv.innerText = "Prediction: " + result.prediction;
            resultDiv.style.color = "green";
        }
    });
});
