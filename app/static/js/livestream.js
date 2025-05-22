const socket = io();
 
 // livestream.js
 const cameraSource = document.getElementById('cameraSelect');

const startCameraButton = document.getElementById('start-camera');
const stopCameraButton = document.getElementById('stop-camera');
const liveVideoContainer = document.getElementById('live-video-container');
let stream = null;

async function populateCameraOptions() {
    try {
        await navigator.mediaDevices.getUserMedia({ video: true });
        const devices = await navigator.mediaDevices.enumerateDevices();
        cameraSource.innerHTML = '<option value="">Select a camera...</option>';
        let index = 0;
        devices.forEach((device, i) => {
            if (device.kind === 'videoinput') {
                const option = document.createElement('option');
                option.value = index;
                option.text = device.label || `Camera ${index + 1}`;
                cameraSource.appendChild(option);
                index++;
            }
        });
    } catch (err) {
        console.error('Camera permission error:', err);
        showAlert('Camera access required', 'danger');
    }
}

startCameraButton.addEventListener('click', async () => {
    const selectedCameraIndex = cameraSource.value;
    if (!selectedCameraIndex && selectedCameraIndex !== 0) return showAlert('Select a camera device', 'warning');

    const constraints = {
        video: {
            deviceId: { exact: selectedCameraIndex },
            width: { ideal: 1280 },
            height: { ideal: 720 }
        }
    };
    try {
        stream = await navigator.mediaDevices.getUserMedia(constraints);
        const video = document.createElement('video');
        video.id = 'live-video-feed';
        video.autoplay = true;
        video.playsInline = true;
        video.srcObject = stream;
        liveVideoContainer.innerHTML = '';
        liveVideoContainer.appendChild(video);
        socket.emit('start_live_processing', { cameraIndex: selectedCameraIndex });

        startCameraButton.disabled = true;
        stopCameraButton.disabled = false;
        cameraSource.disabled = true;
    } catch (e) {
        showAlert('Error accessing camera', 'danger');
    }
});

stopCameraButton.addEventListener('click', () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    liveVideoContainer.innerHTML = '<p class="text-center text-muted">Select a camera source to begin</p>';
    socket.emit('stop_live_processing');
    startCameraButton.disabled = false;
    stopCameraButton.disabled = true;
    cameraSource.disabled = false;
});

// Handle live stream frames
socket.on('frame', (data) => {
    const videoFeed = document.getElementById('live-video-feed');
    if (videoFeed) {
        const blob = new Blob([data.frame], { type: 'image/jpeg' });
        const url = URL.createObjectURL(blob);
        videoFeed.src = url;
    }
});

socket.on("new_event", function(data) {
    const tbody = document.getElementById("live-events-table-body");
    if (!tbody) return;

    const row = document.createElement("tr");
    row.innerHTML = `
        <td>${data.id ?? 'N/A'}</td>
        <td>${data.timestamp}</td>
        <td>${data.event_type}</td>
        <td>${data.vehicle_id}</td>
        <td>${data.motion_status}</td>
        <td>${data.ttc}</td>
        <td>${data.latitude.toFixed(5)}, ${data.longitude.toFixed(5)}</td>
    `;

    tbody.prepend(row);  // Add to top of the table
});

socket.on('gps_update', (data) => {
    console.log("📡 GPS Data Received:", data);
    updateGPSData(data);  // Always update regardless of tab
});

function updateGPSData(data) {
    if (!gpsData) {
        console.error('GPS Data element not found!');
        return;
    }
    if (data.connected) {
        gpsData.innerHTML = `
            <div class="text-success mb-2">GPS Connected</div>
            <div>Latitude: ${data.latitude.toFixed(6)}</div>
            <div>Longitude: ${data.longitude.toFixed(6)}</div>
        `;
    } else {
        gpsData.innerHTML = `
            <div class="text-danger mb-2">GPS Disconnected</div>
            <div>Waiting for GPS signal...</div>
        `;
    }
}
