import os
import requests
from packaging.version import Version
import subprocess
import sys
import traceback
import shutil
import tkinter as tk
from tkinter import messagebox
import stat
import webbrowser

_root = None

def _get_root():
    global _root
    if _root is None:
        _root = tk.Tk()
        _root.title("SmartQSM")
        _root.geometry("1x1+0+0")
        _root.update_idletasks()
    return _root


def _with_topmost(func, title, message):
    root = _get_root()
    root.lift()
    root.attributes("-topmost", True)
    root.update()
    try:
        return func(title, message, parent=root)
    finally:
        # 恢复置顶状态，避免影响其他窗口
        root.attributes("-topmost", False)


def showinfo(title, message):
    return _with_topmost(messagebox.showinfo, title, message)

def showwarning(title, message):
    return _with_topmost(messagebox.showwarning, title, message)

def showerror(title, message):
    return _with_topmost(messagebox.showerror, title, message)

def askyesno(title, message):
    return _with_topmost(messagebox.askyesno, title, message)

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")

def copy_tree_overwrite(src, dst):
    if not os.path.exists(dst):
        os.makedirs(dst, exist_ok=True)

    skip_dirs = {'.git'}  
    for root, dirs, files in os.walk(src):
        
        dirs[:] = [d for d in dirs if d not in skip_dirs]

        rel_path = os.path.relpath(root, src)
        dst_root = os.path.join(dst, rel_path) if rel_path != "." else dst

        os.makedirs(dst_root, exist_ok=True)

        for f in files:
            src_file = os.path.join(root, f)
            dst_file = os.path.join(dst_root, f)
            shutil.copy2(src_file, dst_file)

        for d in dirs:
            src_dir = os.path.join(root, d)
            dst_dir = os.path.join(dst_root, d)
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir, exist_ok=True)

def _get_local_version(path: str) -> Version:
    version_str = "0.0.0"
    try:
        with open(path, "r", encoding="utf-8") as f:
            version_str = f.read().strip()
    except Exception:
        pass
    return Version(version_str)

def _get_remote_version(url: str, timeout: int = 5) -> Version:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    version_str = response.text.strip()
    return Version(version_str)


def _check_update(gui: bool = True) -> int:
    local_version = _get_local_version(os.path.join(ROOT_DIR, "version.txt"))
    
    try:
        remote_version = _get_remote_version("https://raw.githubusercontent.com/project-lightlin/SmartQSM/refs/heads/main/version.txt")
    except Exception as e:
        message = f"Failed to check for updates. The local version is {local_version}. Please visit https://github.com/project-lightlin/SmartQSM yourself or try it later."
        if gui:
            showwarning("Warning", message)
        else:
            print(message)
        return 0
    
    if remote_version <= local_version:
        return 0
    
    message = f"Local version is {local_version} and new version {remote_version} is available. Do you want to update?"

    if gui:
        if not askyesno("Info", message):
            return 0
    else:
        print(message)
        answer = input("[YES/NO]").strip().lower()
        if answer == "no":
            return 0
        elif answer != "yes":
            print("Invalid input. Skipped.\n")
            return 0
    
    # check git
    try:
        subprocess.run(
            ["git", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check="True",
            encoding="utf-8"
        )
    except FileNotFoundError:
        message = f"Git is not installed. Please visit https://git-scm.com/ and install it."
        if gui:
            showerror("Error", message)
            return 1
        else:
            print(message)
            return 1
    except Exception as e:
        message = f"{traceback.format_exc()}\nBad Git. Please visit https://git-scm.com/ and reinstall it." 
        if gui:
            showerror("Error", message)
            return 1
        else:
            print(message)
            return 1
    
    # Preparation
    temp_dir = os.path.abspath(os.path.join(ROOT_DIR, "$temp/"))
    post_installer_path = os.path.join(ROOT_DIR, "post_installation.py")
    if os.path.exists(temp_dir):
        try:
            def handle_remove_readonly(func, path, exc_info):
                try:
                    os.chmod(path, stat.S_IWRITE)
                except Exception:
                    pass
                try:
                    func(path)
                except Exception:
                    pass
            shutil.rmtree(temp_dir, onexc=handle_remove_readonly)
            if os.path.exists(temp_dir):
                raise Exception
        except Exception:
            message = f"Unable to delete directory '$temp/', please try manually deleting and retry."
            if gui:
                showerror("Error", message)
                return 1
            else:
                print(message)
                return 1
    if os.path.exists(post_installer_path):
        try:
            os.remove(post_installer_path)
        except Exception:
            message = f"Unable to delete file 'post_installation.py', please try manually deleting and retry."
            if gui:
                showerror("Error", message)
                return 1
            else:
                print(message)
                return 1

    # Clone
    try:
        cmd = ["git","clone", "https://github.com/project-lightlin/SmartQSM.git", temp_dir]
        p = subprocess.Popen(
            cmd,
            text=True,
            encoding="utf-8"
        )
        p.wait()
        if p.returncode != 0:
            raise subprocess.CalledProcessError(p.returncode, cmd, "")
    except Exception as e:
        message = f"{traceback.format_exc()}\nFailed to clone SmartQSM repository. Network Connection Exception. Please try it later."
        if gui:
            showerror("Error", message)
            return 1
        else:
            print(message)
            return 1

    # Confirm
    message = "The update has been downloaded. Please confirm that there are no other running instances before upgrading!"
    if gui:
        if not askyesno("Info", message):
            return 1
    else:
        print(message)
        answer = input("[YES/NO]").strip().lower()
        if answer == "no":
            return 1
        elif answer != "yes":
            print("Invalid input. Skipped.\n")
            return 1

    # Move
    try:
        copy_tree_overwrite(temp_dir, ROOT_DIR)
    except Exception:
        try:
            os.remove(os.path.join(ROOT_DIR, "version.txt")) # Prevent updating version.txt first, but not all files are successfully transferred. Force in the next upgrade.
        except Exception:
            message = f"{traceback.format_exc()}\nEncountered a serious error during the upgrade process. These errors damage the integrity of the program and may cause various accidents during use. Please visit https://github.com/project-lightlin/SmartQSM to download and reinstall the latest version."
            if gui:
                showerror("Fatal error", message)
                return 1
            else:
                print("\033[31mFatal error: " + message+"\033[0m")
                return 1
        message = f"{traceback.format_exc()}\nFailed to clone SmartQSM repository. Network Connection Exception. Please try it later."
        if gui:
            showerror("Error", message)
            return 1
        else:
            print(message)
            return 1

    # Execute post-installation script
    if os.path.exists(post_installer_path):
        try:
            subprocess.run(
                [sys.executable, post_installer_path], 
                check=True,
                text=True,
                encoding="utf-8"
            )
        except Exception as e:
            try:
                os.remove(post_installer_path)
                os.remove(os.path.join(ROOT_DIR, "version.txt")) # Prevent updating version.txt first, but not all files are successfully transferred. Force in the next upgrade.
            except Exception:
                message = f"{traceback.format_exc()}\nEncountered a serious error during the upgrade process. These errors damage the integrity of the program and may cause various accidents during use. Please visit https://github.com/project-lightlin/SmartQSM to download and reinstall the latest version."
                if gui:
                    showerror("Fatal error", message)
                    return 1
                else:
                    print("\033[31mFatal error: " + message+"\033[0m")
                    return 1
            message = f"{traceback.format_exc()}\nFailed to run post-installation script. Please restart the program."
            if gui:
                showerror("Error", message)
                return 1
            else:
                print(message)
                return 1
            
    # Clean
    try:
        shutil.rmtree(temp_dir, onexc=handle_remove_readonly)
    except Exception:
        pass
    try:
        os.remove(post_installer_path)
    except Exception:
        pass
    
    # Congratulations
    local_version = _get_local_version(os.path.join(ROOT_DIR, "version.txt"))

    message = f"SmartQSM has been successfully updated to version {local_version}! Do you want to know what has been updated?"
    if gui:
        if askyesno("Info", message):
            webbrowser.open("https://github.com/project-lightlin/SmartQSM")
    else:
        print(message)
        answer = input("[Type OK to visit or any to skip]").strip().lower()
        if answer == "ok":
            webbrowser.open("https://github.com/project-lightlin/SmartQSM")
    return 1

def check_update(gui: bool = True) -> None:
    if _check_update(gui) == 1:
        exit(0)
    if _root is not None:
        _root.destroy()