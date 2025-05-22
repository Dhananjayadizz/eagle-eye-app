// utils.js
function showAlert(message, type) {
    let alertContainer = document.getElementById('alertContainer');
    if (!alertContainer) {
        alertContainer = document.createElement('div');
        alertContainer.id = 'alertContainer';
        // Style for top-right fixed positioning, below navbar
        alertContainer.style.position = 'fixed';
        alertContainer.style.top = '70px'; // Adjust this value if your navbar height is different
        alertContainer.style.right = '20px';
        alertContainer.style.zIndex = '1050'; // Ensure it's above other content, like modals
        alertContainer.style.maxWidth = '300px'; // Limit width for better appearance
        // Add some padding or margin to space out multiple alerts
        alertContainer.style.padding = '10px';
        alertContainer.style.display = 'flex'; // Use flexbox to stack alerts vertically
        alertContainer.style.flexDirection = 'column';
        alertContainer.style.gap = '10px'; // Space between alerts

        document.body.appendChild(alertContainer);
    }

    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`; // Added dismissible and fade show classes
    alertDiv.setAttribute('role', 'alert');
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;

    // Prepend the new alert so the newest is at the top
    alertContainer.prepend(alertDiv);

    // Automatically remove alert after a few seconds
    setTimeout(() => {
        // Use Bootstrap's close method if available, otherwise just remove
        const bsAlert = bootstrap.Alert.getInstance(alertDiv) || new bootstrap.Alert(alertDiv);
        bsAlert.close();
    }, 5000); // 5 seconds

    // Optional: Limit the number of alerts shown at once
    const maxAlerts = 5;
    while (alertContainer.children.length > maxAlerts) {
        alertContainer.lastChild.remove();
    }
}


document.addEventListener('DOMContentLoaded', () => {
    stopCameraButton.disabled = true;

    const tabElList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tab"]'));
    tabElList.forEach(tabEl => {
        new bootstrap.Tab(tabEl);
    });

    populateCameraOptions();
});