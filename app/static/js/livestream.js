// const socket = io();
 
//  // livestream.js
//  const cameraSource = document.getElementById('cameraSelect');

// const startCameraButton = document.getElementById('start-camera');
// const stopCameraButton = document.getElementById('stop-camera');
// const liveVideoContainer = document.getElementById('live-video-container');

// const video = document.getElementById('local-video-feed');
// const processedImg = document.getElementById('processed-video-feed');

// let stream = null;

// async function populateCameraOptions() {
//     try {
//         await navigator.mediaDevices.getUserMedia({ video: true });
//         const devices = await navigator.mediaDevices.enumerateDevices();
//         cameraSource.innerHTML = '<option value="">Select a camera...</option>';
//         let index = 0;
//         devices.forEach((device, i) => {
//             if (device.kind === 'videoinput') {
//                 const option = document.createElement('option');
//                 option.value = index;
//                 option.text = device.label || `Camera ${index + 1}`;
//                 cameraSource.appendChild(option);
//                 index++;
//             }
//         });
//     } catch (err) {
//         console.error('Camera permission error:', err);
//         showAlert('Camera access required', 'danger');
//     }
// }

// // startCameraButton.addEventListener('click', async () => {
// //     const selectedCameraIndex = cameraSource.value;
// //     if (!selectedCameraIndex && selectedCameraIndex !== 0) return showAlert('Select a camera device', 'warning');

// //     const constraints = {
// //         video: {
// //             deviceId: { exact: selectedCameraIndex },
// //             width: { ideal: 1280 },
// //             height: { ideal: 720 }
// //         }
// //     };
// //     try {
// //         stream = await navigator.mediaDevices.getUserMedia(constraints);
// //         const video = document.createElement('video');
// //         video.id = 'live-video-feed';
// //         video.autoplay = true;
// //         video.playsInline = true;
// //         video.srcObject = stream;
// //         liveVideoContainer.innerHTML = '';
// //         liveVideoContainer.appendChild(video);
// //         socket.emit('start_live_processing', { cameraIndex: selectedCameraIndex });

// //         startCameraButton.disabled = true;
// //         stopCameraButton.disabled = false;
// //         cameraSource.disabled = true;
// //     } catch (e) {
// //         showAlert('Error accessing camera', 'danger');
// //     }
// // });


// async function startCamera() {
//     try {
//       const stream = await navigator.mediaDevices.getUserMedia({ video: true });
//       video.srcObject = stream;
  
//       const canvas = document.createElement('canvas');
//       const ctx = canvas.getContext('2d');
  
//       video.addEventListener('play', () => {
//         canvas.width = video.videoWidth;
//         canvas.height = video.videoHeight;
  
//         function sendFrame() {
//           if (video.paused || video.ended) return;
  
//           ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  
//           canvas.toBlob(blob => {
//             const reader = new FileReader();
//             reader.onloadend = () => {
//               socket.emit('client_frame', reader.result); // send base64 JPEG frame to backend
//             };
//             reader.readAsDataURL(blob);
//           }, 'image/jpeg', 0.7);
  
//           setTimeout(sendFrame, 33); // ~30 FPS
//         }
  
//         sendFrame();
//       });
//     } catch (err) {
//       console.error('Error accessing camera:', err);
//       alert('Camera access required.');
//     }
//   }
  
//   // Listen for backend processed frames
//   socket.on('processed_frame', (base64Frame) => {
//     processedImg.src = base64Frame;
//   });
  
//   // Start camera on page load or button click
//   startCamera();

// stopCameraButton.addEventListener('click', () => {
//     if (stream) {
//         stream.getTracks().forEach(track => track.stop());
//         stream = null;
//     }
//     liveVideoContainer.innerHTML = '<p class="text-center text-muted">Select a camera source to begin</p>';
//     socket.emit('stop_live_processing');
//     startCameraButton.disabled = false;
//     stopCameraButton.disabled = true;
//     cameraSource.disabled = false;
// });

// // Handle live stream frames
// socket.on('frame', (data) => {
//     const videoFeed = document.getElementById('live-video-feed');
//     if (videoFeed) {
//         const blob = new Blob([data.frame], { type: 'image/jpeg' });
//         const url = URL.createObjectURL(blob);
//         videoFeed.src = url;
//     }
// });

// socket.on("new_event", function(data) {
//     const tbody = document.getElementById("live-events-table-body");
//     if (!tbody) return;

//     const row = document.createElement("tr");
//     row.innerHTML = `
//         <td>${data.id ?? 'N/A'}</td>
//         <td>${data.timestamp}</td>
//         <td>${data.event_type}</td>
//         <td>${data.vehicle_id}</td>
//         <td>${data.motion_status}</td>
//         <td>${data.ttc}</td>
//         <td>${data.latitude.toFixed(5)}, ${data.longitude.toFixed(5)}</td>
//     `;

//     tbody.prepend(row);  // Add to top of the table
// });

// socket.on('gps_update', (data) => {
//     console.log("📡 GPS Data Received:", data);
//     updateGPSData(data);  // Always update regardless of tab
// });

// function updateGPSData(data) {
//     if (!gpsData) {
//         console.error('GPS Data element not found!');
//         return;
//     }
//     if (data.connected) {
//         gpsData.innerHTML = `
//             <div class="text-success mb-2">GPS Connected</div>
//             <div>Latitude: ${data.latitude.toFixed(6)}</div>
//             <div>Longitude: ${data.longitude.toFixed(6)}</div>
//         `;
//     } else {
//         gpsData.innerHTML = `
//             <div class="text-danger mb-2">GPS Disconnected</div>
//             <div>Waiting for GPS signal...</div>
//         `;
//     }
// }


const socket = io();

const cameraSource = document.getElementById('cameraSelect');
const startCameraButton = document.getElementById('start-camera');
const stopCameraButton = document.getElementById('stop-camera');
const liveVideoContainer = document.getElementById('live-video-container');

const video = document.getElementById('local-video-feed');
const processedImg = document.getElementById('processed-video-feed');

let stream = null;

// Populate cameras with actual deviceId strings
async function populateCameraOptions() {
    try {
        await navigator.mediaDevices.getUserMedia({ video: true });
        const devices = await navigator.mediaDevices.enumerateDevices();
        cameraSource.innerHTML = '<option value="">Select a camera...</option>';
        devices.forEach(device => {
            if (device.kind === 'videoinput') {
                const option = document.createElement('option');
                option.value = device.deviceId; // Use actual deviceId here
                option.text = device.label || 'Camera';
                cameraSource.appendChild(option);
            }
        });
    } catch (err) {
        console.error('Camera permission error:', err);
        showAlert('Camera access required', 'danger');
    }
}

async function startCamera() {
    const selectedDeviceId = cameraSource.value;
    if (!selectedDeviceId) {
        alert('Please select a camera device.');
        return;
    }

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                deviceId: { exact: selectedDeviceId },
                width: { ideal: 1280 },
                height: { ideal: 720 }
            }
        });
        video.srcObject = stream;

        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        video.addEventListener('play', () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            function sendFrame() {
                if (video.paused || video.ended) return;

                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

                canvas.toBlob(blob => {
                    const reader = new FileReader();
                    reader.onloadend = () => {
                        socket.emit('client_frame', reader.result); // send base64 JPEG frame to backend
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

        socket.emit('start_live_processing', { cameraIndex: selectedDeviceId });

    } catch (err) {
        console.error('Error accessing camera:', err);
        alert('Camera access required.');
    }
}

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

// Remove 'frame' event handler since processed frames are handled via 'processed_frame'
/*
socket.on('frame', (data) => {
    // Not needed if using 'processed_frame' and <img>
});
*/

// Receive processed frames from backend
socket.on('processed_frame', (base64Frame) => {
    processedImg.src = base64Frame;
});

// Handle incoming events table update
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

    tbody.prepend(row);
});

// GPS data display (ensure you have an element with id 'gpsData')
const gpsData = document.getElementById('gpsData');

socket.on('gps_update', (data) => {
    console.log("📡 GPS Data Received:", data);
    updateGPSData(data);
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

socket.on('processed_frame', (base64Frame) => {
    console.log('Received processed frame');
    const processedImg = document.getElementById('processed-video-feed');
    if (processedImg) processedImg.src = base64Frame;
});


// Initialize cameras dropdown on page load
// populateCameraOptions();


// document.addEventListener('DOMContentLoaded', () => {
//     const socket = io();
  
//     const cameraSource = document.getElementById('cameraSelect');
//     const startCameraButton = document.getElementById('start-camera');
//     const stopCameraButton = document.getElementById('stop-camera');
//     const liveVideoContainer = document.getElementById('live-video-container');
//     const processedImg = document.getElementById('processed-video-feed');
  
//     let stream = null;
//     let video = null;
  
//     async function populateCameraOptions() {
//       try {
//         await navigator.mediaDevices.getUserMedia({ video: true });
//         const devices = await navigator.mediaDevices.enumerateDevices();
//         cameraSource.innerHTML = '<option value="">Select a camera...</option>';
//         let count = 0;
//         devices.forEach(device => {
//           if (device.kind === 'videoinput') {
//             const option = document.createElement('option');
//             option.value = device.deviceId;
//             option.text = device.label || `Camera ${++count}`;
//             cameraSource.appendChild(option);
//           }
//         });
//         if (count === 0) {
//           alert('No camera devices found');
//           startCameraButton.disabled = true;
//         } else {
//           startCameraButton.disabled = false;
//         }
//       } catch (err) {
//         console.error('Camera permission error:', err);
//         alert('Camera access required.');
//         startCameraButton.disabled = true;
//       }
//     }
  
//     if (!video) {
//       video = document.createElement('video');
//       video.style.display = 'none';
//       document.body.appendChild(video);
  
//       video.addEventListener('play', () => {
//         const canvas = document.createElement('canvas');
//         const ctx = canvas.getContext('2d');
//         canvas.width = video.videoWidth;
//         canvas.height = video.videoHeight;
  
//         function sendFrame() {
//           if (video.paused || video.ended) return;
  
//           ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  
//           canvas.toBlob(blob => {
//             const reader = new FileReader();
//             reader.onloadend = () => {
//               socket.emit('client_frame', reader.result);
//             };
//             reader.readAsDataURL(blob);
//           }, 'image/jpeg', 0.7);
  
//           setTimeout(sendFrame, 33); // ~30 FPS
//         }
//         sendFrame();
//       });
  
//       video.addEventListener('error', e => {
//         console.error('Video element error:', e);
//         alert('Error accessing video stream.');
//       });
//     }
  
//     async function startCamera() {
//       const selectedDeviceId = cameraSource.value;
//       if (!selectedDeviceId) {
//         alert('Please select a camera device.');
//         return;
//       }
  
//       if (stream) {
//         stream.getTracks().forEach(track => track.stop());
//         stream = null;
//       }
  
//       try {
//         stream = await navigator.mediaDevices.getUserMedia({
//           video: {
//             deviceId: { exact: selectedDeviceId },
//             width: { ideal: 1280 },
//             height: { ideal: 720 }
//           }
//         });
  
//         video.srcObject = stream;
//         video.play();
  
//         startCameraButton.disabled = true;
//         stopCameraButton.disabled = false;
//         cameraSource.disabled = true;
  
//         socket.emit('start_live_processing', { cameraIndex: selectedDeviceId });
//       } catch (err) {
//         console.error('Error accessing camera:', err);
//         alert('Camera access required.');
//       }
//     }
  
//     function stopCamera() {
//       if (stream) {
//         stream.getTracks().forEach(track => track.stop());
//         stream = null;
//       }
//       if (video) {
//         video.pause();
//         video.srcObject = null;
//       }
//       liveVideoContainer.innerHTML = '<p class="text-center text-muted">Select a camera source to begin</p>';
//       socket.emit('stop_live_processing');
//       startCameraButton.disabled = false;
//       stopCameraButton.disabled = true;
//       cameraSource.disabled = false;
//     }
  
//     socket.on('processed_frame', base64Frame => {
//       if (processedImg) processedImg.src = base64Frame;
//     });
  
//     socket.on('new_event', data => {
//       const tbody = document.getElementById('live-events-table-body');
//       if (!tbody) return;
  
//       const row = document.createElement('tr');
//       row.innerHTML = `
//         <td>${data.id ?? 'N/A'}</td>
//         <td>${data.timestamp}</td>
//         <td>${data.event_type}</td>
//         <td>${data.vehicle_id}</td>
//         <td>${data.motion_status}</td>
//         <td>${data.ttc}</td>
//         <td>${data.latitude.toFixed(5)}, ${data.longitude.toFixed(5)}</td>
//       `;
//       tbody.prepend(row);
//     });
  
//     socket.on('gps_update', data => {
//       updateGPSData(data);
//     });
  
//     function updateGPSData(data) {
//       const gpsData = document.getElementById('gpsData');
//       if (!gpsData) {
//         console.error('GPS Data element not found!');
//         return;
//       }
//       if (data.connected) {
//         gpsData.innerHTML = `
//           <div class="text-success mb-2">GPS Connected</div>
//           <div>Latitude: ${data.latitude.toFixed(6)}</div>
//           <div>Longitude: ${data.longitude.toFixed(6)}</div>
//         `;
//       } else {
//         gpsData.innerHTML = `
//           <div class="text-danger mb-2">GPS Disconnected</div>
//           <div>Waiting for GPS signal...</div>
//         `;
//       }
//     }
  
//     startCameraButton.addEventListener('click', startCamera);
//     stopCameraButton.addEventListener('click', stopCamera);
  
//     populateCameraOptions();
//   });
  