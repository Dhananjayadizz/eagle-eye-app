# blockchain.py
import base64
import json
import bz2
from io import BytesIO
from flask import Blueprint, jsonify, request, send_file
from flask_cors import CORS
from web3 import Web3
import traceback

blockchain_bp = Blueprint('blockchain', __name__)

WEB3_PROVIDER = "http://127.0.0.1:7545"
web3 = Web3(Web3.HTTPProvider(WEB3_PROVIDER))

if not web3.is_connected():
    print("❌ ERROR: Web3 connection failed. Ensure Ganache is running.")

CONTRACT_ADDRESS = "0xB1B5Ce3A53a47e25Cd2dEb60Ff95eD8Bd41983Ae"
SENDER_ACCOUNT = web3.eth.accounts[0]

# Check if contract code exists at the address
contract_code = web3.eth.get_code(CONTRACT_ADDRESS)
if not contract_code or contract_code == b'0x':
    print(f"❌ ERROR: No contract code found at address {CONTRACT_ADDRESS}. Please check the contract address and ensure it is correctly deployed on the network at {WEB3_PROVIDER}.")
else:
    print(f"✅ Contract code found at address {CONTRACT_ADDRESS}.")

CONTRACT_ABI = json.loads("""[
  {
    "inputs": [],
    "name": "getFileNames",
    "outputs": [
      { "internalType": "string[]", "name": "", "type": "string[]" },
      { "internalType": "uint256[]", "name": "", "type": "uint256[]" }
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [
      { "internalType": "uint256", "name": "_fileId", "type": "uint256" }
    ],
    "name": "getZipFile",
    "outputs": [
      { "internalType": "string", "name": "", "type": "string" },
      { "internalType": "uint256", "name": "", "type": "uint256" }
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [
      { "internalType": "string", "name": "_fileName", "type": "string" },
      { "internalType": "string", "name": "_zipData", "type": "string" }
    ],
    "name": "storeZipFile",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  }
]""")

contract = web3.eth.contract(address=CONTRACT_ADDRESS, abi=CONTRACT_ABI)


def compress_bz2(file_data):
    """Compress file data using BZ2."""
    return bz2.compress(file_data)


def decompress_bz2(compressed_data):
    """Decompress BZ2 data."""
    return bz2.decompress(compressed_data)


@blockchain_bp.route("/store", methods=["POST"])
def store():
    try:
        print("📥 Received upload request")
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        file_name = file.filename
        original_data = file.read()
        compressed_data = compress_bz2(original_data)
        encoded_data = base64.b64encode(compressed_data).decode("utf-8")

        print("Uploading file:", file_name)
        print("Original size:", len(original_data))
        print("Compressed size:", len(compressed_data))
        print("Encoded size:", len(encoded_data))
        print("Sender:", SENDER_ACCOUNT)

        # Estimate gas before sending transaction
        try:
            # Changed to use storeZipFile based on the current ABI
            gas_estimate = contract.functions.storeZipFile(file_name, encoded_data).estimate_gas({
                "from": SENDER_ACCOUNT
            })
            print(f"Estimated gas: {gas_estimate}")
            # Add 20% buffer to gas estimate
            gas_limit = int(gas_estimate * 1.2)
        except Exception as gas_error:
            print(f"Gas estimation failed: {str(gas_error)}")
            # Fallback to a reasonable gas limit if estimation fails
            gas_limit = 5000000  # Increased default gas limit

        print(f"Using gas limit: {gas_limit}")

        # Build and send transaction
        # Changed to use storeZipFile based on the current ABI
        tx_hash = contract.functions.storeZipFile(file_name, encoded_data).transact({
            "from": SENDER_ACCOUNT,
            "gas": gas_limit
        })
        
        # Wait for transaction receipt and check status
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        if receipt['status'] == 0:
            print("Transaction failed!")
            # Try to get more detailed error information
            # Note: Getting revert reason from transaction receipt is complex and depends on client support
            # The basic status check is the most reliable way with general clients like Ganache
            print(f"Transaction failed with status 0. Tx hash: {tx_hash.hex()}")
            raise Exception("Transaction failed - check Ganache logs for details")

        print(f"Transaction successful! Block number: {receipt['blockNumber']}")
        return jsonify({"message": "File stored on blockchain using BZ2", "tx_hash": tx_hash.hex()}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@blockchain_bp.route("/list", methods=["GET"])
def list_files():
    try:
        # Changed to use getFileNames based on the current ABI
        # getFileNames returns two arrays: filenames and timestamps
        file_names, timestamps = contract.functions.getFileNames().call({"from": SENDER_ACCOUNT})
        file_list = []
        for i in range(len(file_names)):
             # Assuming fileId is the index + 1, matching previous logic attempt
            file_list.append({
                "id": i + 1, 
                "file_name": file_names[i], 
                "timestamp": timestamps[i] * 1000  # convert to milliseconds
            })
        print(f"✅ Retrieved from Blockchain: {len(file_list)} files")
        file_list.sort(key=lambda x: x['timestamp'], reverse=True)
        return jsonify({"files": file_list}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@blockchain_bp.route("/retrieve/<int:file_id>", methods=["GET"])
def retrieve(file_id):
    try:
        # Changed to use getZipFile based on the current ABI
        # getZipFile returns zipData and timestamp
        # We need to adjust for 1-based indexing if contract expects it, or use 0-based if it expects index.
        # Based on the ABI of getZipFile taking _fileId, it likely expects the ID used in getFileNames.
        # Assuming getZipFile takes the 1-based index as _fileId.
        encoded_data, timestamp = contract.functions.getZipFile(file_id).call({"from": SENDER_ACCOUNT})
        
        # Need to retrieve the filename separately if it's not returned by getZipFile.
        # The original code used getFileNames().call() to get file_names, we can reuse that.
        file_names, _ = contract.functions.getFileNames().call({"from": SENDER_ACCOUNT})
        
        if file_id > len(file_names) or file_id <= 0:
             return jsonify({"error": "Invalid file ID"}), 400
             
        file_name = file_names[file_id - 1] # Adjust for 0-based indexing of the list

        print(f"📥 Retrieving file from blockchain: {file_name} ({len(encoded_data)} encoded characters)")

        compressed_data = base64.b64decode(encoded_data)
        decompressed_data = decompress_bz2(compressed_data)

        print(f"✅ Decompressed data size: {len(decompressed_data)}")

        return send_file(BytesIO(decompressed_data), download_name=file_name, as_attachment=True)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500