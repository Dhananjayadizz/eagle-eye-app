// main.js
// Purpose: Entry point - initializes tabs and default settings

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
