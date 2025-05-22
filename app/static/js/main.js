// main.js
// Purpose: Entry point - initializes tabs and default settings

// document.addEventListener('DOMContentLoaded', () => {
//     stopCameraButton.disabled = true;

//     // Initialize Bootstrap tabs
//     const tabElList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tab"]'));
//     tabElList.forEach(tabEl => {
//         new bootstrap.Tab(tabEl);
//     });

//     // Populate camera options on load
//     populateCameraOptions();

//     // Auto-fetch blockchain files if tab is visible by default
//     if (document.querySelector('#blockchain-tab').classList.contains('active')) {
//         fetchBlockchainFiles();
//     }
// });

document.addEventListener('DOMContentLoaded', () => {
    const stopCameraButton = document.getElementById('stop-camera');
    if(stopCameraButton) stopCameraButton.disabled = true;

    const blockchainTab = document.querySelector('#blockchain-tab');
    if(blockchainTab && blockchainTab.classList.contains('active')) {
        fetchBlockchainFiles();
    }

    const tabElList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tab"]'));
    tabElList.forEach(tabEl => {
        new bootstrap.Tab(tabEl);
    });

    populateCameraOptions();
});
