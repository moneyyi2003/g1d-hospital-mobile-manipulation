#!/usr/bin/env python3
"""Download g1d-expert code from 14.17.59.253 via SFTP."""
import os
import stat
import sys

try:
    import paramiko
except ImportError:
    print("Installing paramiko...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "paramiko", "-q"])
    import paramiko

HOST = "14.17.59.253"
USER = "MaChuanhao"
PASS = "machuanhao"
REMOTE = "/data/MaChuanhao/Project/G1D/g1d-expert"
LOCAL = "/data/MaMingyi/robot-vln/g1d-expert-MaChuanhao"

os.makedirs(LOCAL, exist_ok=True)

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
print(f"Connecting to {USER}@{HOST} ...")
ssh.connect(HOST, username=USER, password=PASS)
sftp = ssh.open_sftp()

file_count = [0]
total_bytes = [0]

def download_dir(remote_dir: str, local_dir: str):
    os.makedirs(local_dir, exist_ok=True)
    for entry in sftp.listdir_attr(remote_dir):
        rpath = f"{remote_dir}/{entry.filename}"
        lpath = os.path.join(local_dir, entry.filename)
        if stat.S_ISDIR(entry.st_mode):
            rel = rpath[len(REMOTE):]
            print(f"  DIR  {rel}/")
            download_dir(rpath, lpath)
        else:
            rel = rpath[len(REMOTE):]
            sftp.get(rpath, lpath)
            file_count[0] += 1
            total_bytes[0] += entry.st_size
            print(f"  FILE {rel}  ({entry.st_size:,} bytes)")

download_dir(REMOTE, LOCAL)
sftp.close()
ssh.close()
print(f"\nDone: {file_count[0]} files, {total_bytes[0]:,} bytes → {LOCAL}")
