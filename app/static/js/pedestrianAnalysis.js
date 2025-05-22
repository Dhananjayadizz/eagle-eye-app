 // pedestrianAnalysis.js
const pedestrianUploadForm = document.getElementById('pedestrian-upload-form');
const pedestrianVideoInput = document.getElementById('pedestrian-video-input');
const pedestrianVideoContainer = document.getElementById('pedestrian-video-container');
const pedestrianUploadLoadingBar = document.getElementById('pedestrian-upload-loading-bar');
const pedestrianProgressBar = pedestrianUploadLoadingBar.querySelector('.progress-bar');

let pedestrianEvents = [];

pedestrianUploadForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const file = pedestrianVideoInput.files[0];
    if (!file) return showAlert('Please select a video file', 'warning');

    const formData = new FormData();
    formData.append('video', file);

    pedestrianUploadLoadingBar.style.display = 'block';
    pedestrianProgressBar.style.width = '0%';

    fetch('/upload_pedestrian_video', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showAlert('Video uploaded successfully', 'success');
            pedestrianVideoContainer.innerHTML = '';
            const videoElement = document.createElement('video');
            videoElement.src = data.video_url;
            videoElement.controls = true;
            videoElement.className = 'w-100';
            pedestrianVideoContainer.appendChild(videoElement);
        } else {
            showAlert(data.error, 'danger');
        }
    })
    .catch(() => showAlert('Error uploading video', 'danger'))
    .finally(() => pedestrianUploadLoadingBar.style.display = 'none');
});

function addPedestrianEvent(event) {
    pedestrianEvents.unshift(event);
    updatePedestrianEventsTable();
}

function updatePedestrianEventsTable() {
    const table = document.getElementById('pedestrian-events-table');
    table.innerHTML = pedestrianEvents.map(event => `
        <tr class="${getPedestrianEventRowClass(event)}">
            <td>${event.id}</td>
            <td>${event.timestamp}</td>
            <td>${event.pedestrian_id}</td>
            <td>${event.intent_score.toFixed(2)}</td>
            <td>${event.speed.toFixed(2)} px/frame</td>
            <td>${getPedestrianStatus(event.intent_score)}</td>
            <td>(${event.location.x}, ${event.location.y})</td>
        </tr>
    `).join('');
}

function getPedestrianEventRowClass(event) {
    if (event.intent_score >= 0.7) return 'table-danger';
    if (event.intent_score >= 0.5) return 'table-warning';
    return 'table-success';
}

function getPedestrianStatus(score) {
    return score >= 0.7 ? 'High Risk' : score >= 0.5 ? 'Medium Risk' : 'Low Risk';
}