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
    if (!file) {
        showAlert('Please select a file to upload.', 'warning');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    blockchainUploadStatus.textContent = 'Uploading...';
    blockchainUploadButton.disabled = true;

    try {
        const response = await fetch('/blockchain/store', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            blockchainUploadStatus.textContent = data.message;
            showAlert('File uploaded successfully!', 'success');
            fetchBlockchainFiles();
        } else {
            blockchainUploadStatus.textContent = `Error: ${data.error}`;
            showAlert(`Error uploading file: ${data.error}`, 'danger');
        }
    } catch (error) {
        console.error('Error uploading file:', error);
        blockchainUploadStatus.textContent = 'Error uploading file.';
        showAlert('Error uploading file.', 'danger');
    } finally {
        blockchainUploadButton.disabled = false;
        blockchainFileInput.value = '';
    }
});

blockchainRefreshButton.addEventListener('click', fetchBlockchainFiles);

document.querySelector('#blockchain-tab').addEventListener('shown.bs.tab', () => {
    fetchBlockchainFiles();
});

async function fetchBlockchainFiles() {
    blockchainListStatus.textContent = 'Loading files...';
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
                    <td>${new Date(file.timestamp * 1000).toLocaleString()}</td>
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
        } else {
            blockchainFilesTable.innerHTML = '<tr><td colspan="4">No files found on the blockchain.</td></tr>';
            blockchainListStatus.textContent = '';
        }
    } catch (error) {
        console.error('Error fetching blockchain files:', error);
        blockchainListStatus.textContent = 'Error loading files.';
        blockchainFilesTable.innerHTML = '<tr><td colspan="4">Could not load files.</td></tr>';
    }
}
