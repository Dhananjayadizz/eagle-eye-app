// blockchain.js
// Purpose: Upload and manage files stored via the blockchain backend

const blockchainFileInput = document.getElementById('blockchain-file-input');
const blockchainUploadButton = document.getElementById('blockchain-upload-button');
const blockchainUploadStatus = document.getElementById('blockchain-upload-status');
const blockchainRefreshButton = document.getElementById('blockchain-refresh-button');
const blockchainFilesTable = document.getElementById('blockchain-files-table');
const blockchainListStatus = document.getElementById('blockchain-list-status');

blockchainUploadButton.addEventListener('click', async () => {
    const file = blockchainFileInput.files[0];
    blockchainUploadStatus.textContent = '';
    if (!file) {
        blockchainUploadStatus.textContent = 'Please select a file to upload.';
        blockchainUploadStatus.className = 'text-warning mb-3';
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    blockchainUploadStatus.textContent = 'Uploading...';
    blockchainUploadStatus.className = 'text-info mb-3';
    blockchainUploadButton.disabled = true;

    try {
        const response = await fetch('/blockchain/store', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            blockchainUploadStatus.textContent = 'File stored on blockchain using BZ2';
            blockchainUploadStatus.className = 'text-success mb-3';
            fetchBlockchainFiles();
        } else {
            blockchainUploadStatus.textContent = `Error: ${data.error}`;
            blockchainUploadStatus.className = 'text-danger mb-3';
        }
    } catch (error) {
        console.error('Error uploading file:', error);
        blockchainUploadStatus.textContent = 'Error uploading file.';
        blockchainUploadStatus.className = 'text-danger mb-3';
    } finally {
        blockchainUploadButton.disabled = false;
        blockchainFileInput.value = '';
    }
});

blockchainRefreshButton.addEventListener('click', fetchBlockchainFiles);

// If you use tabs, you may want to auto-refresh when the tab is shown
const blockchainTab = document.querySelector('#blockchain-tab');
if (blockchainTab) {
    blockchainTab.addEventListener('shown.bs.tab', () => {
        fetchBlockchainFiles();
    });
}

document.addEventListener('DOMContentLoaded', () => {
    fetchBlockchainFiles();
});

function formatTimestamp(ts) {
    // ts is in milliseconds
    const date = new Date(ts);
    const options = {
        year: 'numeric', month: 'numeric', day: 'numeric',
        hour: 'numeric', minute: '2-digit', second: '2-digit',
        hour12: true
    };
    return date.toLocaleString(undefined, options);
}

async function fetchBlockchainFiles() {
    blockchainListStatus.textContent = 'Loading files...';
    blockchainListStatus.className = 'text-info mb-2';
    try {
        const response = await fetch('/blockchain/list');
        const data = await response.json();

        blockchainFilesTable.innerHTML = '';

        if (data.files && data.files.length > 0) {
            data.files.forEach(file => {
                const row = blockchainFilesTable.insertRow();
                row.innerHTML = `
                    <td>${file.id}</td>
                    <td>${file.file_name}</td>
                    <td>${formatTimestamp(file.timestamp)}</td>
                    <td><button class="btn btn-sm btn-success download-btn" data-file-id="${file.id}">Download</button></td>
                `;
            });
            document.querySelectorAll('.download-btn').forEach(button => {
                button.addEventListener('click', (e) => {
                    const fileId = e.target.dataset.fileId;
                    window.location.href = `/blockchain/retrieve/${fileId}`;
                });
            });
            blockchainListStatus.textContent = '';
            blockchainListStatus.className = 'mb-2';
        } else {
            blockchainFilesTable.innerHTML = '<tr><td colspan="4">No files found on the blockchain.</td></tr>';
            blockchainListStatus.textContent = '';
            blockchainListStatus.className = 'mb-2';
        }
    } catch (error) {
        console.error('Error fetching blockchain files:', error);
        blockchainListStatus.textContent = 'Error loading files.';
        blockchainListStatus.className = 'text-danger mb-2';
        blockchainFilesTable.innerHTML = '<tr><td colspan="4">Could not load files.</td></tr>';
    }
}
