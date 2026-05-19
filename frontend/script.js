const API_URL = "http://localhost:8000/predict";

const form = document.getElementById("diagnosticForm");
const resultDiv = document.getElementById("resultat");

let historique = [];

form.addEventListener("submit", async (e) => {

    e.preventDefault();

    const data = {
        age: parseInt(document.getElementById("age").value),
        sexe: document.getElementById("sexe").value,
        temperature: parseFloat(document.getElementById("temperature").value),
        tension_sys: parseInt(document.getElementById("tension").value),
        region: document.getElementById("region").value,
        toux: document.getElementById("toux").checked,
        fatigue: document.getElementById("fatigue").checked,
        maux_tete: document.getElementById("maux_tete").checked
    };

    try {

        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        let emoji = "🩺";

        if (result.diagnostic === "palu") {
            emoji = "🌡️";
        }
        else if (result.diagnostic === "grippe") {
            emoji = "🤧";
        }
        else if (result.diagnostic === "typh") {
            emoji = "💊";
        }
        else if (result.diagnostic === "sain") {
            emoji = "✅";
        }

        resultDiv.innerHTML = `
            <h2>Résultat</h2>

            <p><strong>Diagnostic :</strong> ${emoji} ${result.diagnostic}</p>

            <p><strong>Probabilité :</strong> ${result.probabilite}</p>

            <p><strong>Confiance :</strong> ${result.confiance}</p>

            <p><strong>Message :</strong> ${result.message}</p>
        `;

        historique.unshift({
            nom: data.region,
            diagnostic: result.diagnostic,
            probabilite: result.probabilite
        });

        historique = historique.slice(0, 5);

        afficherHistorique();

    }

    catch (error) {

        resultDiv.innerHTML = `
            <p style="color:red;">
                Erreur de connexion à l'API
            </p>
        `;
    }

});

function afficherHistorique() {

    let historiqueHTML = `
        <h2>Historique des consultations</h2>

        <table>

            <tr>
                <th>Nom</th>
                <th>Diagnostic</th>
                <th>Probabilité</th>
            </tr>
    `;

    historique.forEach(item => {

        historiqueHTML += `
            <tr>
                <td>${item.nom}</td>
                <td>${item.diagnostic}</td>
                <td>${item.probabilite}</td>
            </tr>
        `;
    });

    historiqueHTML += `</table>`;

    document.getElementById("historique").innerHTML = historiqueHTML;
}