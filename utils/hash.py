import hashlib
import sys

def get_md5_hash(file_path):
    with open(file_path, 'rb') as file_obj:
        digest = hashlib.file_digest(file_obj, 'md5')  # Use 'md5' for MD5
    return digest.hexdigest()