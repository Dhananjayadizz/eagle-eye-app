 // gps.js
const gpsData = document.getElementById('gps-data');

function updateGPSData(data) {
    if (!gpsData) return;
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