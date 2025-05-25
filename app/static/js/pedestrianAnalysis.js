// pedestrianAnalysis.js
const socket = io();  // Initialize Socket.IO connection

// DOM Elements
const pedestrianVideoInput = document.getElementById('pedestrian-video-input');
const pedestrianUploadForm = document.getElementById('pedestrian-upload-form');
const pedestrianVideoContainer = document.getElementById('pedestrian-video-container');
const pedestrianEventsTable = document.getElementById('pedestrian-events-table');
const pedestrianUploadLoadingBar = document.getElementById('pedestrian-upload-loading-bar');
const pedestrianProgressBar = pedestrianUploadLoadingBar.querySelector('.progress-bar');

// State variables
let pedestrianEvents = [];
let currentVideoId = null;
let isProcessing = false;

// Handle video upload
pedestrianUploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const file = pedestrianVideoInput.files[0];
    if (!file) {
        showAlert('Please select a video file', 'warning');
        return;
    }

    if (!file.type.startsWith('video/')) {
        showAlert('Please select a valid video file', 'warning');
        return;
    }

    if (isProcessing) {
        showAlert('Video is currently being processed', 'warning');
        return;
    }

    const formData = new FormData();
    formData.append('video', file);

    // Show loading bar and reset progress
    pedestrianUploadLoadingBar.style.display = 'block';
    pedestrianProgressBar.style.width = '0%';
    pedestrianProgressBar.setAttribute('aria-valuenow', '0');
    isProcessing = true;

    const xhr = new XMLHttpRequest();

    // Progress event listener
    xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
            const percentComplete = (event.loaded / event.total) * 100;
            pedestrianProgressBar.style.width = percentComplete + '%';
            pedestrianProgressBar.setAttribute('aria-valuenow', percentComplete);
        }
    };

    // Load event listener (upload complete)
    xhr.onload = () => {
        pedestrianUploadLoadingBar.style.display = 'none';
        if (xhr.status === 200) {
            const data = JSON.parse(xhr.responseText);
            if (data.error) {
                showAlert(data.error, 'danger');
                isProcessing = false;
            } else {
                currentVideoId = data.video_id;
                showAlert('Video uploaded successfully! Starting analysis...', 'success');
                
                // Display video feed immediately after upload
                displayPedestrianVideoFeed();
                
                // Automatically start processing after successful upload
                setTimeout(() => {
                    socket.emit('start_pedestrian_analysis', { video_id: currentVideoId });
                }, 500); // Small delay to ensure video feed is established
            }
        } else {
            let errorMessage = 'Error uploading video. ';
            try {
                const data = JSON.parse(xhr.responseText);
                errorMessage += data.error || `Status: ${xhr.status}`;
            } catch (e) {
                errorMessage += `Status: ${xhr.status}`;
            }
            showAlert(errorMessage, 'danger');
            isProcessing = false;
        }
    };

    // Error event listener
    xhr.onerror = () => {
        pedestrianUploadLoadingBar.style.display = 'none';
        showAlert('Network error during video upload. Please check your connection and try again.', 'danger');
        isProcessing = false;
    };

    // Abort event listener
    xhr.onabort = () => {
        pedestrianUploadLoadingBar.style.display = 'none';
        showAlert('Video upload was cancelled.', 'warning');
        isProcessing = false;
    };

    xhr.open('POST', '/upload_pedestrian_video');
    xhr.send(formData);
});

// Display the uploaded video feed
function displayPedestrianVideoFeed() {
    pedestrianVideoContainer.innerHTML = '';

    // Create a container for the video feed with a loading message
    const container = document.createElement('div');
    container.className = 'position-relative';
    
    // Add the image element for the MJPEG stream
    const img = document.createElement('img');
    img.src = `/pedestrian_video_feed?t=${Date.now()}`; // Add timestamp to prevent caching
    img.className = 'w-100';
    img.alt = 'Pedestrian video feed';
    img.onerror = () => {
        showAlert('Error loading video feed. Please try again.', 'danger');
    };
    
    container.appendChild(img);
    pedestrianVideoContainer.appendChild(container);
    
    // Log that video feed is being displayed
    console.log('Displaying pedestrian video feed');
}

// Add pedestrian event to the list
function addPedestrianEvent(event) {
    pedestrianEvents.unshift(event);
    updatePedestrianEventsTable();
}

// Update the pedestrian events table
function updatePedestrianEventsTable() {
    if (!pedestrianEventsTable) return;
    
    pedestrianEventsTable.innerHTML = pedestrianEvents.slice().reverse().map(event => {
        const intentScore = typeof event.intent_score === 'number' ? (event.intent_score * 100).toFixed(1) + '%' : 'N/A';
        const speed = typeof event.speed === 'number' ? event.speed.toFixed(2) + ' px/frame' : 'N/A';
        const location = event.location ? `(${event.location.x.toFixed(1)}, ${event.location.y.toFixed(1)})` : 'N/A';
        
        return `
            <tr class="${getPedestrianEventRowClass(event)}">
                <td>${formatTimestamp(event.timestamp)}</td>
                <td>${event.pedestrian_id || 'N/A'}</td>
                <td>${intentScore}</td>
                <td>${speed}</td>
                <td>${getPedestrianStatus(event.intent_score)}</td>
                <td>${location}</td>
            </tr>
        `;
    }).join('');
}

// Determine row class based on intent score
function getPedestrianEventRowClass(event) {
    let rowClass = 'custom-row';
    
    if (event.intent_score >= 0.7) {
        rowClass += ' border-left-red'; // High risk - red
    } else if (event.intent_score >= 0.5) {
        rowClass += ' border-left-orange'; // Medium risk - orange
    } else {
        rowClass += ' border-left-green'; // Low risk - green
    }
    
    return rowClass;
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
    if (!timestamp) return 'N/A';
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
}

// Socket.IO Event Handler for New Pedestrian Events
socket.on('pedestrian_event', (event) => {
    // Add event to the list
    pedestrianEvents.push(event);
    
    // Update the table
    const targetTableBody = pedestrianEventsTable;
    
    if (targetTableBody) {
        const row = document.createElement('tr');
        let rowClass = 'custom-row';
        
        // Determine left border color class based on intent score
        if (event.intent_score >= 0.7) {
            rowClass += ' border-left-red'; // High risk - red
        } else if (event.intent_score >= 0.5) {
            rowClass += ' border-left-orange'; // Medium risk - orange
        } else {
            rowClass += ' border-left-green'; // Low risk - green
        }
        
        row.className = rowClass;
        
        // Format values
        const timestamp = formatTimestamp(event.timestamp);
        const intentScore = typeof event.intent_score === 'number' ? (event.intent_score * 100).toFixed(1) + '%' : 'N/A';
        const speed = typeof event.speed === 'number' ? event.speed.toFixed(2) + ' px/frame' : 'N/A';
        const location = event.location ? `(${event.location.x.toFixed(1)}, ${event.location.y.toFixed(1)})` : 'N/A';
        const status = getPedestrianStatus(event.intent_score);
        
        // Add event data to row
        row.innerHTML = `
            <td>${timestamp}</td>
            <td>${event.pedestrian_id || 'N/A'}</td>
            <td>${intentScore}</td>
            <td>${speed}</td>
            <td>${status}</td>
            <td>${location}</td>
        `;
        
        targetTableBody.insertBefore(row, targetTableBody.firstChild);
        
        // Keep only last 100 events
        if (targetTableBody.children.length > 100) {
            targetTableBody.removeChild(targetTableBody.lastChild);
        }
    }
});

// Socket event for analysis completion
socket.on('analysis_complete', () => {
    isProcessing = false;
    showAlert('Video analysis completed', 'success');
});

// Socket event for errors
socket.on('error', (data) => {
    showAlert(data.message || 'An error occurred during processing', 'danger');
    isProcessing = false;
});

// Export pedestrian events
function exportPedestrianEvents() {
    if (pedestrianEvents.length === 0) {
        showAlert('No pedestrian events to export', 'warning');
        return;
    }
    window.location.href = '/export_pedestrian_events';
}

// Clear pedestrian files
function clearPedestrianFiles() {
    if (confirm('Are you sure you want to clear all pedestrian files?')) {
        fetch('/clear_pedestrian_files', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showAlert(data.error, 'danger');
            } else {
                showAlert('All pedestrian files cleared successfully', 'success');
                pedestrianEvents = [];
                updatePedestrianEventsTable();
                pedestrianVideoContainer.innerHTML = '<p class="text-center text-muted">Upload a video to see the analysis here.</p>';
            }
        })
        .catch(error => {
            showAlert('Error clearing files: ' + error.message, 'danger');
        });
    }
}

// Show alert message
function showAlert(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.textContent = message;

    const container = document.querySelector('.container');
    if (container) {
        container.insertBefore(alertDiv, container.firstChild);
        setTimeout(() => alertDiv.remove(), 5000);
    } else {
        alert(message); // fallback
    }
}

// Initialize event listeners when document is loaded
document.addEventListener('DOMContentLoaded', function() {
    const exportButton = document.getElementById('export-pedestrian-events-btn');
    if (exportButton) {
        exportButton.addEventListener('click', exportPedestrianEvents);
    }

    const clearButton = document.getElementById('clear-pedestrian-files-btn');
    if (clearButton) {
        clearButton.addEventListener('click', clearPedestrianFiles);
    }
});
