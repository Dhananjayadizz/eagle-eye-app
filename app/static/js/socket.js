 // socket.js
const socket = io();

socket.on('connect', () => console.log('Connected to server'));
socket.on('disconnect', () => showAlert('Connection lost. Please refresh the page.', 'danger'));
socket.on('error', (data) => showAlert(data.message, 'danger'));
socket.on('gps_update', updateGPSData);
socket.on('frame', (data) => {
    const videoFeed = document.getElementById('live-video-feed');
    if (videoFeed) {
        const blob = new Blob([data.frame], { type: 'image/jpeg' });
        videoFeed.src = URL.createObjectURL(blob);
    }
});
socket.on('pedestrian_event', (event) => addPedestrianEvent(event));