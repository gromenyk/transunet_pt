// app.js

// Obtain progress from the inference
function pollProgress() {
    console.log("Polling...");
    fetch('/progress')
        .then(response => response.json())
        .then(data => {
            const progressBarFill = document.getElementById('progress-bar-fill')
            const progressMessage = document.getElementById('progress-message');
            const progressText = document.getElementById('progress-text');
            const progressTime = document.getElementById('progress-time');
            const progressBackground = document.getElementById('.progress-bar div')

            // Actualiza la barra de progreso
            progressBackground.style.width = data.percent + '%';
            progressBarFill.textContent = data.percent + '%';

            // Actualiza el tiempo transcurrido
            if (!isNaN(data.elapsed_time)) {
                const minutes = Math.floor(data.elapsed_time / 60);
                const seconds = data.elapsed_time % 60;
                progressTime.innerText = `⏱️ Elapsed Time: ${minutes}m ${seconds}s`;
            } else {
                progressTime.innerText = ""; // En caso de que el tiempo sea inválido
            }
            
            // Muestra el mensaje actual
            progressMessage.textContent = data.message;

            // Si el progreso está en curso, sigue actualizando cada 1 segundo
            if (data.status === 'running') {
                setTimeout(pollProgress, 1000);
            }
        })
        .catch(error => console.error('Error en la solicitud de progreso:', error));
}

document.querySelector('form[action="/run_inference"]').addEventListener('submit', function(event) {
    event.preventDefault();
    fetch('/run_inference', {
        method: 'POST',
    })
    .then(response => response.text())  // Espera la respuesta del servidor (la página renderizada)
    .then(data => {
        // Iniciar el polling para actualizar los logs
        setInterval(pollProgress, 1000);
    })
    .catch(error => console.error('Error en la solicitud de inferencia:', error));
});

// Obtain filename from the video selection to display it on UI
const fileInput = document.getElementById('video');
const fileNameSpan = document.getElementById('file-name');
const uploadButton = document.getElementById('upload-button');
const uploadMessage = document.getElementById('upload-message')

fileInput.addEventListener('change', function () {
    if (this.files.length > 0) {
        fileNameSpan.textContent = this.files[0].name;
        uploadButton.style.display = 'inline-block';

        if (uploadMessage) {
            uploadMessage.style.display = 'none'
        }
        
    } else {
        fileNameSpan.textContent = '';
        uploadButton.style.display = 'none'
    }
});
