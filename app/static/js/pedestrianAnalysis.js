// pedestrianAnalysis.js
const socket = io()

// Pedestrian Analysis Variables
const pedestrianVideoInput = document.getElementById('pedestrian-video-input');
const pedestrianUploadForm = document.getElementById('pedestrian-upload-form');
const pedestrianVideoContainer = document.getElementById('pedestrian-video-container');
const pedestrianEventsTable = document.getElementById('pedestrian-events-table');
const pedestrianUploadLoadingBar = document.getElementById('pedestrian-upload-loading-bar');
const pedestrianProgressBar = pedestrianUploadLoadingBar.querySelector('.progress-bar');

let pedestrianEvents = [];
let isProcessing = false;

pedestrianUploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const file = pedestrianVideoInput.files[0];
    if (!file) {
        showAlert('Please select a video file', 'warning');
        return;
    }

    if (isProcessing) {
        showAlert('Video is currently being processed', 'warning');
        return;
    }

    const formData = new FormData();
    formData.append('video', file);

    pedestrianUploadLoadingBar.style.display = 'block';
    pedestrianProgressBar.style.width = '0%';
    isProcessing = true;

    try {
        const response = await fetch('/upload_pedestrian_video', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (data.success && data.video_url) {
            showAlert('Video uploaded successfully', 'success');
            pedestrianVideoContainer.innerHTML = '';
            
            const videoElement = document.createElement('video');
            videoElement.id = 'pedestrian-video-feed';
            videoElement.src = `/uploads/pedestrian_video.mp4?t=${Date.now()}`;
            videoElement.controls = true;
            videoElement.autoplay = true;
            videoElement.muted = true;
            videoElement.className = 'w-100';

            pedestrianVideoContainer.appendChild(videoElement);
            
            // Start real-time analysis
            startPedestrianAnalysis();
        } else {
            showAlert(data.error || 'Failed to upload video', 'danger');
        }
    } catch (error) {
        console.error('Error:', error);
        showAlert('Error uploading video', 'danger');
    } finally {
        pedestrianUploadLoadingBar.style.display = 'none';
        isProcessing = false;
    }
});

// Socket event handlers
socket.on('pedestrian_event', (event) => {
    addPedestrianEvent(event);
});

socket.on('analysis_complete', () => {
    isProcessing = false;
    showAlert('Video analysis completed', 'success');
});

// Add pedestrian event to the list
function addPedestrianEvent(event) {
    pedestrianEvents.unshift(event);
    updatePedestrianEventsTable();
}

// Update the pedestrian events table
function updatePedestrianEventsTable() {
    if (!pedestrianEventsTable) return;
    
    pedestrianEventsTable.innerHTML = pedestrianEvents.map(event => `
        <tr class="${getPedestrianEventRowClass(event)}">
            <td>${event.id}</td>
            <td>${formatTimestamp(event.timestamp)}</td>
            <td>${event.pedestrian_id}</td>
            <td>${(event.intent_score * 100).toFixed(1)}%</td>
            <td>${event.speed.toFixed(2)} px/frame</td>
            <td>${getPedestrianStatus(event.intent_score)}</td>
            <td>(${event.location.x.toFixed(1)}, ${event.location.y.toFixed(1)})</td>
        </tr>
    `).join('');
}

// Determine row class based on intent score
function getPedestrianEventRowClass(event) {
    if (event.intent_score >= 0.7) {
        return 'table-danger'; // Red for high intent
    } else if (event.intent_score >= 0.5) {
        return 'table-warning'; // Orange for medium intent
    }
    return 'table-success'; // Green for low intent
}

// Get pedestrian status based on intent score
function getPedestrianStatus(intentScore) {
    if (intentScore >= 0.7) {
        return 'High Risk';
    } else if (intentScore >= 0.5) {
        return 'Medium Risk';
    }
    return 'Low Risk';
}

// Format timestamp to readable format
function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
}

// Start pedestrian analysis
function startPedestrianAnalysis() {
    socket.emit('start_pedestrian_analysis');
}

// Export pedestrian events
function exportPedestrianEvents() {
    if (pedestrianEvents.length === 0) {
        showAlert('No pedestrian events to export', 'warning');
        return;
    }
    window.location.href = '/export_pedestrian_events';
}

// Clear pedestrian files
async function clearPedestrianFiles() {
    try {
        const response = await fetch('/clear_pedestrian_files', {
            method: 'POST',
        });
        const data = await response.json();
        
        if (data.success) {
            showAlert('Pedestrian files cleared successfully', 'success');
            pedestrianEvents = [];
            updatePedestrianEventsTable();
            pedestrianVideoContainer.innerHTML = '<p class="text-center text-muted">Upload a video to see the analysis here.</p>';
        } else {
            showAlert('Failed to clear pedestrian files', 'danger');
        }
    } catch (error) {
        console.error('Error:', error);
        showAlert('Error clearing pedestrian files', 'danger');
    }
}