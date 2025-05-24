// trafficanalysis.js
document.addEventListener('DOMContentLoaded', function() {
    const socket = io();  // Initialize Socket.IO connection
    
    const uploadForm = document.getElementById('videoUploadForm');
    const videoInput = document.getElementById('videoFile');
    const uploadedVideoContainer = document.getElementById('uploaded-video-container');
    const processingStatus = document.getElementById('processingStatus');
    const exportBtn = document.getElementById('exportBtn');
    const clearFilesBtn = document.getElementById('clearFilesBtn');
    const eventsTableBody = document.getElementById('eventsTableBody');
    
    let trafficEvents = [];
    
    // Handle video upload
    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = new FormData(this);
        const videoFile = document.getElementById('videoFile').files[0];
        
        if (!videoFile) {
            showAlert('Please select a video file', 'warning');
            return;
        }
        
        // Show processing status
        processingStatus.style.display = 'block';
        
        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                showAlert('Video uploaded successfully', 'success');
                displayUploadedVideoFeed();
                exportBtn.disabled = false;
            } else {
                showAlert(data.error || 'Failed to upload video', 'danger');
            }
        } catch (error) {
            console.error('Error:', error);
            showAlert('Error uploading video', 'danger');
        } finally {
            // Hide processing status
            processingStatus.style.display = 'none';
        }
    });
    
    // Function to display the video feed
    function displayUploadedVideoFeed() {
        uploadedVideoContainer.innerHTML = '';
        
        const img = document.createElement('img');
        img.src = `/video_feed?t=${Date.now()}`; // Add timestamp to prevent caching
        img.className = 'w-100';
        uploadedVideoContainer.appendChild(img);
    }
    
    // Handle export button
    exportBtn.addEventListener('click', async function() {
        try {
            const response = await fetch('/export_events');
            const blob = await response.blob();
            
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'traffic_detection_events.xlsx';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } catch (error) {
            showAlert('Error exporting events: ' + error.message, 'danger');
        }
    });

    // Handle clear files button
    clearFilesBtn.addEventListener('click', async function() {
        try {
            const response = await fetch('/clear_files', {
                method: 'POST'
            });
            const data = await response.json();
            
            if (data.success) {
                showAlert('Files cleared successfully', 'success');
                trafficEvents = [];
                eventsTableBody.innerHTML = '';
                uploadedVideoContainer.innerHTML = 
                    '<p class="text-center text-muted">Upload a video to see the analysis here.</p>';
                exportBtn.disabled = true;
            } else {
                showAlert('Failed to clear files', 'danger');
            }
        } catch (error) {
            showAlert('Error clearing files: ' + error.message, 'danger');
        }
    });
    
    // Socket.IO event handling for new events
    socket.on('new_event', function(event) {
        // Add event to array
        trafficEvents.push(event);
        
        // Keep only last 100 events in memory
        if (trafficEvents.length > 100) {
            trafficEvents.shift();
        }
        
        // Create a new row for the event
        const row = document.createElement('tr');
        
        // Set row class based on event type
        if (event.event_type === 'Pothole') {
            row.className = 'border-left-red';
        } else if (event.event_type === 'Number Plate') {
            row.className = 'border-left-blue';
        } else if (event.event_type.includes('Stop Light Violation')) {
            row.className = 'border-left-yellow';
        }
        
        // Format timestamp
        const timestamp = new Date(event.timestamp).toLocaleTimeString();
        
        // Format TTC
        const ttc = event.ttc === null || event.ttc === Infinity ? 'N/A' : event.ttc.toFixed(2);
        
        // Format location
        const location = `${parseFloat(event.latitude).toFixed(6)}, ${parseFloat(event.longitude).toFixed(6)}`;
        
        row.innerHTML = `
            <td>${event.id || 'N/A'}</td>
            <td>${timestamp}</td>
            <td>${event.event_type}</td>
            <td>${event.vehicle_id || 'N/A'}</td>
            <td>${event.motion_status}</td>
            <td>${ttc}</td>
            <td>${location}</td>
        `;
        
        // Add row to table (at the top)
        eventsTableBody.insertBefore(row, eventsTableBody.firstChild);
        
        // Keep only last 100 rows in the table
        if (eventsTableBody.children.length > 100) {
            eventsTableBody.removeChild(eventsTableBody.lastChild);
        }
    });
    
    // Function to show alerts
    function showAlert(message, type) {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show mt-3`;
        alertDiv.role = 'alert';
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        const container = document.querySelector('.container');
        container.insertBefore(alertDiv, container.firstChild);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            alertDiv.classList.remove('show');
            setTimeout(() => alertDiv.remove(), 300);
        }, 5000);
    }
});
