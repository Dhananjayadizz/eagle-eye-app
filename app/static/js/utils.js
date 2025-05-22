 // utils.js
function showAlert(message, type) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.textContent = message;

    const container = document.querySelector('.container');
    container.insertBefore(alertDiv, container.firstChild);

    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}


document.addEventListener('DOMContentLoaded', () => {
    stopCameraButton.disabled = true;

    const tabElList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tab"]'));
    tabElList.forEach(tabEl => {
        new bootstrap.Tab(tabEl);
    });

    populateCameraOptions();
});