 // livestream.js
const cameraSource = document.getElementById('camera-source');
const startCameraButton = document.getElementById('start-camera');
const stopCameraButton = document.getElementById('stop-camera');
const liveVideoContainer = document.getElementById('live-video-container');
let stream = null;

async function populateCameraOptions() {
    try {
        await navigator.mediaDevices.getUserMedia({ video: true });
        const devices = await navigator.mediaDevices.enumerateDevices();
        cameraSource.innerHTML = '<option value="">Select a camera...</option>';
        devices.forEach((d, i) => {
            if (d.kind === 'videoinput') {
                const option = document.createElement('option');
                option.value = d.deviceId;
                option.text = d.label || `Camera ${i + 1}`;
                cameraSource.appendChild(option);
            }
        });
    } catch (err) {
        console.error('Camera permission error:', err);
        showAlert('Camera access required', 'danger');
    }
}

startCameraButton.addEventListener('click', async () => {
    const deviceId = cameraSource.value;
    if (!deviceId) return showAlert('Select a camera device', 'warning');

    const constraints = {
        video: {
            deviceId: { exact: deviceId },
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
        socket.emit('start_live_processing', { deviceId });

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