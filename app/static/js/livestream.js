// Optimized livestream.js for near real-time performance
const socket = io();

const cameraSource = document.getElementById('cameraSelect');
const startCameraButton = document.getElementById('start-camera');
const stopCameraButton = document.getElementById('stop-camera');
const liveVideoContainer = document.getElementById('live-video-container');
const processedImg = document.getElementById('processed-video-feed');
const loadingIndicator = document.getElementById('video-loading-indicator');
const qualitySelector = document.getElementById('quality-selector');
const fpsSelector = document.getElementById('fps-selector');

let stream = null;
let video = null;
let isProcessing = false;
let frameReceived = false;
let loadingTimeout = null;
let lastFrameTime = 0;
let frameInterval = 100; // Default 10 FPS (100ms)
let frameQuality = 0.5; // Default medium quality
let frameResolution = { width: 640, height: 480 }; // Default medium resolution

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
            showAlert('Video processing is taking longer than expected. Try reducing quality or frame rate.', 'warning');
        }
    }, 10000); // Show warning after 10 seconds if no frames received
}

// Update frame settings based on user selections
function updateFrameSettings() {
    // Update frame rate
    if (fpsSelector) {
        const fps = parseInt(fpsSelector.value);
        frameInterval = Math.floor(1000 / fps);
    }
    
    // Update quality
    if (qualitySelector) {
        const quality = qualitySelector.value;
        switch(quality) {
            case 'low':
                frameQuality = 0.3;
                frameResolution = { width: 320, height: 240 };
                break;
            case 'medium':
                frameQuality = 0.5;
                frameResolution = { width: 640, height: 480 };
                break;
            case 'high':
                frameQuality = 0.7;
                frameResolution = { width: 1280, height: 720 };
                break;
        }
    }
    
    if (isProcessing) {
        showAlert(`Settings updated: ${frameResolution.width}x${frameResolution.height} at ${Math.floor(1000/frameInterval)} FPS`, 'info');
    }
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
        // Update frame settings before starting
        updateFrameSettings();
        
        // Show loading indicator before starting camera
        showLoading();
        isProcessing = true;
        frameReceived = false;
        setLoadingTimeout();
        
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                deviceId: { exact: selectedDeviceId },
                width: { ideal: frameResolution.width },
                height: { ideal: frameResolution.height }
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
        canvas.width = frameResolution.width;
        canvas.height = frameResolution.height;

        video.addEventListener('play', () => {
            function sendFrame() {
                if (video.paused || video.ended || !isProcessing) return;
                
                // Throttle frame rate
                const now = Date.now();
                if (now - lastFrameTime < frameInterval) {
                    setTimeout(sendFrame, 5); // Check again soon
                    return;
                }
                
                lastFrameTime = now;
                
                // Draw at the specified resolution
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

                canvas.toBlob(blob => {
                    const reader = new FileReader();
                    reader.onloadend = () => {
                        socket.emit('client_frame', reader.result);
                    };
                    reader.readAsDataURL(blob);
                }, 'image/jpeg', frameQuality);

                setTimeout(sendFrame, 5); // Schedule next frame check
            }

            sendFrame();
        });

        startCameraButton.disabled = true;
        stopCameraButton.disabled = false;
        cameraSource.disabled = true;
        
        if (qualitySelector) qualitySelector.disabled = false;
        if (fpsSelector) fpsSelector.disabled = false;

        // Notify server to start processing with the selected camera
        socket.emit('start_live_processing', { 
            cameraIndex: 0,
            quality: qualitySelector ? qualitySelector.value : 'medium',
            fps: fpsSelector ? parseInt(fpsSelector.value) : 10
        });
        
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
    
    if (qualitySelector) qualitySelector.disabled = true;
    if (fpsSelector) fpsSelector.disabled = true;
    
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
if (qualitySelector) {
    qualitySelector.addEventListener('change', updateFrameSettings);
}
if (fpsSelector) {
    fpsSelector.addEventListener('change', updateFrameSettings);
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    populateCameraOptions();
    stopCameraButton.disabled = true;
    
    // Disable quality and FPS selectors until camera starts
    if (qualitySelector) qualitySelector.disabled = true;
    if (fpsSelector) fpsSelector.disabled = true;
    
    // Set default placeholder image
    if (processedImg && !processedImg.src) {
        processedImg.src = '/static/images/video-placeholder.jpg';
    }
});
