import base64
import bz2
import json
import traceback
from io import BytesIO

from flask import Blueprint, jsonify, request, send_file
from flask_cors import CORS
from web3 import Web3

blockchain_bp = Blueprint('blockchain', __name__)

WEB3_PROVIDER = "http://127.0.0.1:7545"
web3 = Web3(Web3.HTTPProvider(WEB3_PROVIDER))

print(f"[DEBUG] Web3 provider: {WEB3_PROVIDER}")
print(f"[DEBUG] Web3 connected: {web3.is_connected()}")

CONTRACT_ADDRESS = "0xe9E9e0E7A6cc96A73f32587bCF3A4A4bF84aed38"
print(f"[DEBUG] Contract address: {CONTRACT_ADDRESS}")

try:
    SENDER_ACCOUNT = web3.eth.accounts[0]
    print(f"[DEBUG] Sender account: {SENDER_ACCOUNT}")
except Exception as e:
    print(f"[ERROR] Could not get sender account: {e}")
    SENDER_ACCOUNT = None

# Check if contract code exists at the address
contract_code = web3.eth.get_code(CONTRACT_ADDRESS)
if contract_code == b'' or contract_code == b'0x':
    print(f"❌ ERROR: No contract code found at address {CONTRACT_ADDRESS}.")
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

print(f"[DEBUG] Contract ABI loaded: {len(CONTRACT_ABI)} functions")
contract = web3.eth.contract(address=CONTRACT_ADDRESS, abi=CONTRACT_ABI)


def compress_bz2(file_data):
    return bz2.compress(file_data)


def decompress_bz2(compressed_data):
    return bz2.decompress(compressed_data)


@blockchain_bp.route("/store", methods=["POST"])
def store():
    try:
        print("📥 Received upload request")
        if "file" not in request.files:
            print("[ERROR] No file uploaded in request.files")
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        file_name = file.filename.strip()
        if not file_name:
            return jsonify({"error": "Invalid file name"}), 400

        original_data = file.read()
        if not original_data:
            return jsonify({"error": "Uploaded file is empty"}), 400

        compressed_data = compress_bz2(original_data)
        encoded_data = base64.b64encode(compressed_data).decode("utf-8")

        print(f"Uploading file: {file_name}")
        print(f"Original size: {len(original_data)} bytes")
        print(f"Compressed size: {len(compressed_data)} bytes")
        print(f"Encoded base64 size: {len(encoded_data)} chars")
        print(f"Sender: {SENDER_ACCOUNT}")

        try:
            gas_estimate = contract.functions.storeZipFile(file_name, encoded_data).estimate_gas({
                "from": SENDER_ACCOUNT
            })
            print(f"Estimated gas: {gas_estimate}")
            gas_limit = int(gas_estimate * 1.2)
        except Exception as gas_error:
            print(f"[ERROR] Gas estimation failed: {str(gas_error)}")
            gas_limit = 5000000

        print(f"Using gas limit: {gas_limit}")

        try:
            tx_hash = contract.functions.storeZipFile(file_name, encoded_data).transact({
                "from": SENDER_ACCOUNT,
                "gas": gas_limit
            })
            print(f"[DEBUG] Transaction hash: {tx_hash.hex()}")
            receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
            print(f"[DEBUG] Transaction receipt: {receipt}")
            if receipt['status'] == 0:
                print("[ERROR] Transaction failed! Status 0.")
                raise Exception("Transaction failed - check Ganache logs for details")
        except Exception as tx_error:
            print(f"[ERROR] Transaction error: {tx_error}")
            traceback.print_exc()
            return jsonify({"error": f"Blockchain transaction failed: {tx_error}"}), 500

        print(f"✅ Transaction successful! Block number: {receipt['blockNumber']}")
        return jsonify({"message": "File stored on blockchain using BZ2", "tx_hash": tx_hash.hex()}), 200
    except Exception as e:
        print(f"[ERROR] Exception in /store: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@blockchain_bp.route("/list", methods=["GET"])
def list_files():
    try:
        print("[DEBUG] Fetching file list from blockchain...")
        file_names, timestamps = contract.functions.getFileNames().call({"from": SENDER_ACCOUNT})
        file_list = []
        for i in range(len(file_names)):
            file_list.append({
                "id": i + 1,
                "file_name": file_names[i],
                "timestamp": timestamps[i] * 1000
            })
        print(f"✅ Retrieved from Blockchain: {len(file_list)} files")
        file_list.sort(key=lambda x: x['timestamp'], reverse=True)
        return jsonify({"files": file_list}), 200
    except Exception as e:
        print(f"[ERROR] Exception in /list: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@blockchain_bp.route("/retrieve/<int:file_id>", methods=["GET"])
def retrieve(file_id):
    try:
        print(f"[DEBUG] Retrieving file with ID: {file_id}")
        encoded_data, timestamp = contract.functions.getZipFile(file_id).call({"from": SENDER_ACCOUNT})
        file_names, _ = contract.functions.getFileNames().call({"from": SENDER_ACCOUNT})
        if file_id > len(file_names) or file_id <= 0:
            print(f"[ERROR] Invalid file ID: {file_id}")
            return jsonify({"error": "Invalid file ID"}), 400
        file_name = file_names[file_id - 1]
        print(f"📥 Retrieving file from blockchain: {file_name} ({len(encoded_data)} encoded characters)")
        compressed_data = base64.b64decode(encoded_data)
        decompressed_data = decompress_bz2(compressed_data)
        print(f"✅ Decompressed data size: {len(decompressed_data)}")
        return send_file(BytesIO(decompressed_data), download_name=file_name, as_attachment=True)
    except Exception as e:
        print(f"[ERROR] Exception in /retrieve: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
