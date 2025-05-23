// SPDX-License-Identifier: MIT
pragma solidity >=0.5.0 <0.9.0;
pragma experimental ABIEncoderV2;

contract DataStorage {
    struct FileData {
        string fileName;
        string encodedZipFile;
        uint256 timestamp; // Added timestamp for storing time and date
    }

    mapping(uint256 => FileData) public storedFiles;
    uint256 public fileCount = 1; // Start IDs from 1

    event ZipFileStored(uint256 indexed id, string fileName, uint256 timestamp);

    function storeZipFile(string memory _fileName, string memory _zipData) public {
        storedFiles[fileCount] = FileData(_fileName, _zipData, block.timestamp);
        emit ZipFileStored(fileCount, _fileName, block.timestamp);
        fileCount++;
    }

    function getFileNames() public view returns (string[] memory, uint256[] memory) {
        string[] memory fileNames = new string[](fileCount - 1);
        uint256[] memory timestamps = new uint256[](fileCount - 1);
        
        for (uint256 i = 1; i < fileCount; i++) {
            fileNames[i - 1] = storedFiles[i].fileName;
            timestamps[i - 1] = storedFiles[i].timestamp;
        }
        
        return (fileNames, timestamps);
    }

    function getZipFile(uint256 _fileId) public view returns (string memory, uint256) {
        require(_fileId > 0 && _fileId < fileCount, "Invalid file ID");
        return (storedFiles[_fileId].encodedZipFile, storedFiles[_fileId].timestamp);
    }
}
