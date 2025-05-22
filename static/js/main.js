document.addEventListener('DOMContentLoaded', function() {
    // Initialize Socket.IO
    var socket = io();

    const criticalEventsTable = document.getElementById('critical-events-table');

    // Socket.IO event handler for new events
    socket.on('new_event', (event) => {
        // Add a row for every received event
        const criticalEventsTableBody = document.getElementById('critical-events-table');

        if (criticalEventsTableBody) {
            const row = document.createElement('tr');
            let rowClass = 'custom-row';

            // Color coding based on event severity
            // Keep color coding based on the event data received from the backend
            if (event.motion_status === 'Collided' || event.event_type === 'Near Collision') {
                rowClass += ' border-red';
            } else if (event.event_type && event.event_type.includes('Anomaly')) {
                rowClass += ' border-yellow';
            } else if (event.event_type === 'Frontier') {
                rowClass += ' border-blue';
            }
            row.className = rowClass;

            // Add event data to row
            row.innerHTML = `
                <td>${event.id !== undefined ? event.id : 'N/A'}</td>
                <td>${event.timestamp !== undefined ? event.timestamp : 'N/A'}</td>
                <td>${event.event_type !== undefined ? event.event_type : 'N/A'}</td>
                <td>${event.vehicle_id !== undefined ? event.vehicle_id : 'N/A'}</td>
                <td>${event.motion_status !== undefined ? event.motion_status : 'N/A'}</td>
                <td>${
                    let ttc = Number(event.ttc);
                    let ttcDisplay = isNaN(ttc) ? "N/A" : ttc.toFixed(2);
                    return ttcDisplay;
                }</td>
                <td>${event.latitude !== undefined && event.longitude !== undefined ? `${parseFloat(event.latitude).toFixed(6)}, ${parseFloat(event.longitude).toFixed(6)}` : 'N/A'}</td>
            `;
            criticalEventsTableBody.insertBefore(row, criticalEventsTableBody.firstChild);
        }
    });

    // Add event listener for the export button
    const exportButton = document.getElementById('export-critical-events-btn');
    if (exportButton) {
        exportButton.addEventListener('click', function() {
            window.location.href = '/export_critical_events';
        });
    }
}); 