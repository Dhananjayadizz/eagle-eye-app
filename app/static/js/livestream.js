// Updated livestream.js with loading effect
const socket = io();

const cameraSource = document.getElementById('cameraSelect');
const startCameraButton = document.getElementById('start-camera');
const stopCameraButton = document.getElementById('stop-camera');
const liveVideoContainer = document.getElementById('live-video-container');
const processedImg = document.getElementById('processed-video-feed');
const loadingIndicator = document.getElementById('video-loading-indicator');

let stream = null;
let video = null;
let isProcessing = false;
let frameReceived = false;
let loadingTimeout = null;

// Show loading indicator
function showLoading() {
    if (loadingIndicator) {
        loadingIndicator.style.display = 'flex';
    }
    if (processedImg) {
        processedImg.style.opacity = '0.3';
    }
}

// Hide loading indicator
function hideLoading() {
    if (loadingIndicator) {
        loadingIndicator.style.display = 'none';
    }
    if (processedImg) {
        processedImg.style.opacity = '1';
    }
}

// Set timeout for loading indicator
function setLoadingTimeout() {
    clearTimeout(loadingTimeout);
    loadingTimeout = setTimeout(() => {
        if (!frameReceived && isProcessing) {
            showAlert('Video processing is taking longer than expected. Please be patient or try again.', 'warning');
        }
    }, 10000); // Show warning after 10 seconds if no frames received
}

// Populate camera options
async function populateCameraOptions() {
    try {
        await navigator.mediaDevices.getUserMedia({ video: true });
        const devices = await navigator.mediaDevices.enumerateDevices();
        cameraSource.innerHTML = '<option value="">Select a camera...</option>';
        devices.forEach(device => {
            if (device.kind === 'videoinput') {
                const option = document.createElement('option');
                option.value = device.deviceId;
                option.text = device.label || 'Camera';
                cameraSource.appendChild(option);
            }
        });
    } catch (err) {
        console.error('Camera permission error:', err);
        showAlert('Camera access required. Please allow camera permissions.', 'danger');
    }
}

// Start camera and send frames to server
async function startCamera() {
    const selectedDeviceId = cameraSource.value;
    if (!selectedDeviceId) {
        showAlert('Please select a camera device.', 'warning');
        return;
    }

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    try {
        // Show loading indicator before starting camera
        showLoading();
        isProcessing = true;
        frameReceived = false;
        setLoadingTimeout();
        
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                deviceId: { exact: selectedDeviceId },
                width: { ideal: 1280 },
                height: { ideal: 720 }
            }
        });
        
        if (!video) {
            video = document.createElement('video');
            video.autoplay = true;
            video.style.display = 'none';
            document.body.appendChild(video);
        }
        
        video.srcObject = stream;

        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        video.addEventListener('play', () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            function sendFrame() {
                if (video.paused || video.ended || !isProcessing) return;

                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

                canvas.toBlob(blob => {
                    const reader = new FileReader();
                    reader.onloadend = () => {
                        socket.emit('client_frame', reader.result);
                    };
                    reader.readAsDataURL(blob);
                }, 'image/jpeg', 0.7);

                setTimeout(sendFrame, 33); // ~30 FPS
            }

            sendFrame();
        });

        startCameraButton.disabled = true;
        stopCameraButton.disabled = false;
        cameraSource.disabled = true;

        // Notify server to start processing with the selected camera
        socket.emit('start_live_processing', { cameraIndex: 0 }); // Use 0 for default camera index
        
        showAlert('Camera started. Processing video stream...', 'info');

    } catch (err) {
        console.error('Error accessing camera:', err);
        showAlert('Camera access error: ' + err.message, 'danger');
        hideLoading();
        isProcessing = false;
    }
}

// Stop camera
function stopCamera() {
    isProcessing = false;
    frameReceived = false;
    clearTimeout(loadingTimeout);
    
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    
    if (video) {
        video.pause();
        video.srcObject = null;
    }
    
    startCameraButton.disabled = false;
    stopCameraButton.disabled = true;
    cameraSource.disabled = false;
    
    socket.emit('stop_live_processing');
    
    // Reset the video display
    if (processedImg) {
        processedImg.src = '/static/images/video-placeholder.jpg';
    }
    
    hideLoading();
    showAlert('Camera stopped', 'info');
}

// Receive processed frames from server
socket.on('processed_frame', (base64Frame) => {
    console.log('Received processed frame');
    if (processedImg) {
        // Hide loading indicator on first frame
        if (!frameReceived) {
            frameReceived = true;
            hideLoading();
            clearTimeout(loadingTimeout);
        }
        processedImg.src = base64Frame;
    }
});

// Handle stream errors
socket.on('stream_error', (data) => {
    showAlert('Stream error: ' + data.error, 'danger');
    stopCamera();
});

// Handle status updates
socket.on('status', (data) => {
    console.log('Status update:', data.message);
    showAlert(data.message, 'info');
});

// Handle incoming events table update
socket.on("new_event", function(data) {
    const tbody = document.getElementById("critical-events-table");
    if (!tbody) return;

    const row = document.createElement("tr");
    row.innerHTML = `
        <td>${data.id ?? 'N/A'}</td>
        <td>${data.timestamp}</td>
        <td>${data.event_type}</td>
        <td>${data.vehicle_id}</td>
        <td>${data.motion_status}</td>
        <td>${data.ttc}</td>
        <td>${data.latitude ? data.latitude.toFixed(5) : 'N/A'}, ${data.longitude ? data.longitude.toFixed(5) : 'N/A'}</td>
    `;

    tbody.prepend(row);
});

// Handle connection errors
socket.on('connect_error', (error) => {
    console.error('Connection error:', error);
    showAlert('Connection error: ' + error.message, 'danger');
    if (isProcessing) {
        stopCamera();
    }
});

// Handle disconnection
socket.on('disconnect', (reason) => {
    console.log('Disconnected:', reason);
    if (isProcessing) {
        showAlert('Connection lost. Stopping camera.', 'warning');
        stopCamera();
    }
});

// Event listeners
startCameraButton.addEventListener('click', startCamera);
stopCameraButton.addEventListener('click', stopCamera);

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    populateCameraOptions();
    stopCameraButton.disabled = true;
    
    // Set default placeholder image
    if (processedImg && !processedImg.src) {
        processedImg.src = '/static/images/video-placeholder.jpg';
    }
});
