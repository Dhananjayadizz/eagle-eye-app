; // criticalEvents.js
 const socket = io();  // 👈 this initializes Socket.IO connection


const uploadForm = document.getElementById('videoUploadForm');
const videoInput = document.getElementById('videoFile');
const uploadedVideoContainer = document.getElementById('uploaded-video-container'); // Since <video> is your preview
// const uploadLoadingBar = document.getElementById('upload-loading-bar');
// const uploadProgressBar = document.getElementById('upload-progress-bar');
const criticalEventsTable = document.getElementById('eventsTableBody');




let currentVideoId = null;
let criticalEvents = [];


// Core Critical Event Functions
function addCriticalEvent(event) {
    criticalEvents.unshift(event);
    updateCriticalEventsTable();
}

// function updateCriticalEventsTable() {
//     criticalEventsTable.innerHTML = criticalEvents.map(event => `
//         <tr>
//             <td>${event.timestamp}</td>
//             <td>${event.event_type}</td>
//             <td>${event.vehicle_id}</td>
//             <td>${event.motion_status}</td>
//             <td>${event.ttc !== 'N/A' ? event.ttc + 's' : 'N/A'}</td>
//             <td>${event.latitude.toFixed(6)}, ${event.longitude.toFixed(6)}</td>
//         </tr>
//     `).join('');
// }

// function updateCriticalEventsTable() {
//     criticalEventsTable.innerHTML = criticalEvents.map(event => {
//         const ttc = typeof event.ttc === 'number' ? `${event.ttc.toFixed(2)}s` : 'N/A';
//         const location = `${parseFloat(event.latitude).toFixed(6)}, ${parseFloat(event.longitude).toFixed(6)}`;
//         const isCritical = event.is_critical ? 'Yes' : 'No';
//         return `
//             <tr>
//                 <td>${event.id}</td>
//                 <td>${event.timestamp}</td>
//                 <td>${event.event_type}</td>
//                 <td>${event.vehicle_id}</td>
//                 <td>${event.motion_status}</td>
//                 <td>${ttc}</td>
//                 <td>${location}</td>
//             </tr>
//         `;
//     }).join('');
// }

function updateCriticalEventsTable() {
    criticalEventsTable.innerHTML = criticalEvents.slice().reverse().map(event => {
        const ttc = typeof event.ttc === 'number' ? `${event.ttc.toFixed(2)}s` : 'N/A';
        const latitude = event.latitude !== undefined ? parseFloat(event.latitude).toFixed(6) : 'N/A';
        const longitude = event.longitude !== undefined ? parseFloat(event.longitude).toFixed(6) : 'N/A';
        const location = `${latitude}, ${longitude}`;
        const isCritical = event.is_critical ? 'Yes' : 'No';

        return `
            <tr>
                <td>${event.id}</td>
                <td>${event.timestamp}</td>
                <td>${event.event_type}</td>
                <td>${event.vehicle_id}</td>
                <td>${event.motion_status}</td>
                <td>${ttc}</td>
                <td>${location}</td>
            </tr>
        `;
    }).join('');
}



// Export Functions
function exportCriticalEvents() {
    if (criticalEvents.length === 0) {
        showAlert('No critical events to export', 'warning');
        return;
    }
    window.location.href = '/export_critical_events';
}

function clearExportedFiles() {
    if (confirm('Are you sure you want to clear all exported files?')) {
        fetch('/clear_exported_files', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showAlert(data.error, 'danger');
            } else {
                showAlert('All exported files cleared successfully', 'success');
            }
        })
        .catch(error => {
            showAlert('Error clearing files: ' + error.message, 'danger');
        });
    }
}


function addEventToLog(event) {
    const eventElement = document.createElement('div');
    eventElement.className = 'event-item';
    eventElement.innerHTML = `
        <strong>${event.event_type}</strong>
        <br>
        Vehicle ID: ${event.vehicle_id}
        <br>
        Time: ${event.timestamp}
        <br>
        Status: ${event.motion_status}
        ${event.ttc !== 'N/A' ? `<br>TTC: ${event.ttc}s` : ''}
    `;
    eventsLog.insertBefore(eventElement, eventsLog.firstChild);
}


socket.on('new_event', (event) => {
    criticalEvents.push(event); // Add new event to list
    updateCriticalEventsTable(); // Refresh the table
});


// Socket.IO Event Handler for New Events
socket.on('new_event', (event) => {
    // Add all events to the appropriate real-time analysis table based on active tab
    const criticalTab = document.getElementById('critical-tab');
    const liveTab = document.getElementById('livestream-tab');
    
    let targetTableBody = null;
    if (criticalTab && criticalTab.classList.contains('active') && criticalEventsTable) {
        targetTableBody = criticalEventsTable.querySelector('tbody');
    }
    
    if (targetTableBody) {
        const row = document.createElement('tr');
        let rowClass = 'custom-row';
        
        // Determine row class based on event type and motion status
        if (event.motion_status === 'Collided' || event.event_type === 'Near Collision') {
            rowClass += ' border-red';
        } else if (event.event_type && event.event_type.includes('Anomaly')) {
            rowClass += ' border-yellow';
        } else if (event.event_type === 'Frontier') {
            rowClass += ' border-blue';
        }
        row.className = rowClass;
        
        // Add event data to row
        row.innerHTML = `
            <td>${event.id !== undefined ? event.id : 'N/A'}</td>
            <td>${event.timestamp !== undefined ? event.timestamp : 'N/A'}</td>
            <td>${event.event_type !== undefined ? event.event_type : 'N/A'}</td>
            <td>${event.vehicle_id !== undefined ? event.vehicle_id : 'N/A'}</td>
            <td>${event.motion_status !== undefined ? event.motion_status : 'N/A'}</td>
            <td>${event.ttc !== undefined && event.ttc !== null && event.ttc !== 'N/A' ? parseFloat(event.ttc).toFixed(2) : 'N/A'}</td>
            <td>${event.latitude !== undefined && event.longitude !== undefined ? `${parseFloat(event.latitude).toFixed(6)}, ${parseFloat(event.longitude).toFixed(6)}` : 'N/A'}</td>
        `;
        targetTableBody.insertBefore(row, targetTableBody.firstChild);
    }
});

// socket.on('new_event', (event) => {
//     const criticalTab = document.getElementById('critical-tab');
//     const liveTab = document.getElementById('livestream-tab');

//     let targetTableBody = null;
//     if (criticalTab && criticalTab.classList.contains('active') && criticalEventsTable) {
//         targetTableBody = criticalEventsTable.querySelector('tbody');
//     }

//     if (targetTableBody) {
//         const row = document.createElement('tr');
//         let rowClass = 'custom-row';

//         // ✅ Determine left border color
//         if ('is_critical' in event) {
//             rowClass += event.is_critical ? ' border-left-red' : ' border-left-green';
//         } else {
//             if (event.motion_status === 'Collided' || event.event_type === 'Near Collision') {
//                 rowClass += ' border-left-red';
//             } else if (event.event_type && event.event_type.includes('Anomaly')) {
//                 rowClass += ' border-left-yellow';
//             } else if (event.event_type === 'Frontier') {
//                 rowClass += ' border-left-blue';
//             }
//         }

//         row.className = rowClass;

//         const timestamp = event.timestamp ? new Date(event.timestamp).toLocaleTimeString() : 'N/A';
//         const ttc = event.ttc !== undefined && event.ttc !== null && event.ttc !== 'N/A'
//             ? parseFloat(event.ttc).toFixed(2) : 'N/A';
//         const location = (event.latitude !== undefined && event.longitude !== undefined)
//             ? `${parseFloat(event.latitude).toFixed(6)}, ${parseFloat(event.longitude).toFixed(6)}` : 'N/A';

//         row.innerHTML = `
//             <td>${event.id ?? 'N/A'}</td>
//             <td>${timestamp}</td>
//             <td>${event.event_type ?? 'N/A'}</td>
//             <td>${event.vehicle_id ?? 'N/A'}</td>
//             <td>${event.motion_status ?? 'N/A'}</td>
//             <td>${ttc}</td>
//             <td>${location}</td>
//         `;

//         targetTableBody.insertBefore(row, targetTableBody.firstChild);

//         if (targetTableBody.children.length > 100) {
//             targetTableBody.removeChild(targetTableBody.lastChild);
//         }
//     }
// });







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


// uploadForm.addEventListener('submit', (e) => {
//     e.preventDefault();
//     const file = videoInput.files[0];
//     if (!file || !file.type.startsWith('video/')) {
//         return showAlert('Please select a valid video file.', 'warning');
//     }

//     const formData = new FormData();
//     formData.append('video', file);

//     uploadLoadingBar.style.display = 'block';
//     uploadProgressBar.style.width = '0%';

//     const xhr = new XMLHttpRequest();
//     xhr.upload.onprogress = (event) => {
//         const percent = (event.loaded / event.total) * 100;
//         uploadProgressBar.style.width = `${percent}%`;
//     };
//     xhr.onload = () => {
//         uploadLoadingBar.style.display = 'none';
//         const data = JSON.parse(xhr.responseText);
//         if (data.error) return showAlert(data.error, 'danger');
//         currentVideoId = data.video_id;
//         showAlert('Video uploaded successfully!', 'success');
//         displayUploadedVideoFeed();
//         socket.emit('start_processing', { video_id: currentVideoId });
//     };
//     xhr.onerror = () => showAlert('Upload failed', 'danger');
//     xhr.open('POST', '/upload');
//     xhr.send(formData);
// });

// Handle video upload (in Critical Event Analysis tab)
uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const file = videoInput.files[0];
    if (!file) {
        showAlert('Please select a video file to upload.', 'warning');
        return;
    }

    // Validate file type
    if (!file.type.startsWith('video/')) {
        showAlert('Please select a valid video file.', 'warning');
        return;
    }

    const formData = new FormData();
    formData.append('video', file);

    // Show loading bar and reset progress
    uploadLoadingBar.style.display = 'block';
    uploadProgressBar.style.width = '0%';
    uploadProgressBar.setAttribute('aria-valuenow', '0');

    const xhr = new XMLHttpRequest();

    // Progress event listener
    xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
            const percentComplete = (event.loaded / event.total) * 100;
            uploadProgressBar.style.width = percentComplete + '%';
            uploadProgressBar.setAttribute('aria-valuenow', percentComplete);
        }
    };

    // Load event listener (upload complete)
    xhr.onload = () => {
        uploadLoadingBar.style.display = 'none';
        if (xhr.status === 200) {
            const data = JSON.parse(xhr.responseText);
            if (data.error) {
                showAlert(data.error, 'danger');
            } else {
                currentVideoId = data.video_id;
                showAlert('Video uploaded successfully! Starting analysis...', 'success');
                displayUploadedVideoFeed();
                
                // Automatically start processing after successful upload
                socket.emit('start_processing', { video_id: currentVideoId });
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
        }
    };

    // Error event listener
    xhr.onerror = () => {
        uploadLoadingBar.style.display = 'none';
        showAlert('Network error during video upload. Please check your connection and try again.', 'danger');
    };

    // Abort event listener
    xhr.onabort = () => {
        uploadLoadingBar.style.display = 'none';
        showAlert('Video upload was cancelled.', 'warning');
    };

    xhr.open('POST', '/upload');
    xhr.send(formData);
});


// function displayUploadedVideoFeed() {
//     uploadedVideoContainer.innerHTML = '';
    
//     const video = document.createElement('video');
//     video.id = 'uploaded-video-feed';
//     video.src = `/uploads/latest_video.mp4?t=${Date.now()}`;
//     video.controls = true;
//     video.className = 'video-container';
//     video.autoplay = true;
//     video.muted = true;
    
//     uploadedVideoContainer.appendChild(video);
// }

// function displayUploadedVideoFeed() {
//     uploadedVideoContainer.innerHTML = '';

//     // Add timestamp to force refresh
//     const timestamp = new Date().getTime();
//     const video = document.createElement('img');
//     video.id = 'uploaded-video-feed';
//     video.src = `/video_feed?t=${timestamp}`; // 💡 this avoids caching
//     uploadedVideoContainer.appendChild(video);
// }


// function displayUploadedVideoFeed() {
//     uploadedVideoContainer.innerHTML = '';

//     const video = document.createElement('video');
//     video.src = `/uploads/latest_video.mp4?t=${Date.now()}`;
//     video.controls = true;
//     video.className = 'video-container';
//     uploadedVideoContainer.appendChild(video);
// }


function displayUploadedVideoFeed() {
    uploadedVideoContainer.innerHTML = '';

    const img = document.createElement('img');
    img.src = `/video_feed?t=${Date.now()}`; // live-processed stream
    img.className = 'w-100';
    uploadedVideoContainer.appendChild(img);
}




// if (data.success) {
//     exportBtn.disabled = false;

//     // Show video preview in the new container
//     uploadedVideoContainer.innerHTML = '';
//     const video = document.createElement('video');
//     video.src = `/uploads/latest_video.mp4?t=${Date.now()}`;
//     video.controls = true;
//     video.className = 'w-100';
//     uploadedVideoContainer.appendChild(video);
// }


