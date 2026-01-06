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

    import tempfile
    try:
        worker_code = r'''
import os, shutil, stat, sys, time, traceback, webbrowser, psutil
import tkinter as tk
from tkinter import messagebox

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
        root.attributes("-topmost", False)


def showinfo(title, message):
    return _with_topmost(messagebox.showinfo, title, message)

def showwarning(title, message):
    return _with_topmost(messagebox.showwarning, title, message)

def showerror(title, message):
    return _with_topmost(messagebox.showerror, title, message)

def askyesno(title, message):
    return _with_topmost(messagebox.askyesno, title, message)

def handle_remove_readonly(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
    except Exception:
        pass
    try:
        func(path)
    except Exception:
        pass

def copy_tree_overwrite(src, dst):
    if not os.path.exists(dst):
        os.makedirs(dst, exist_ok=True)
    skip_dirs = {'.git', '$temp'}
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        rel_path = os.path.relpath(root, src)
        dst_root = os.path.join(dst, rel_path) if rel_path != "." else dst
        os.makedirs(dst_root, exist_ok=True)
        for f in files:
            src_file = os.path.join(root, f)
            dst_file = os.path.join(dst_root, f)
            shutil.copy2(src_file, dst_file)

def _has_running_entrypoints() -> bool:
    targets = {
        "smartqsm.py",
        "qsm_viewer.py",
        "parameter_exporter.py",
        "stand_structurer.py"
    }

    current_pid = os.getpid()

    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            pid = proc.info["pid"]
            if pid == current_pid:
                continue

            cmdline = proc.info.get("cmdline") or []
            if not cmdline:
                continue

            for arg in cmdline:
                norm_arg = arg.replace("\\", "/").lower()
                for t in targets:
                    if t in norm_arg:
                        return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    return False

def main():
    if len(sys.argv) != 4:
        sys.exit(1)
    temp_dir = os.path.abspath(sys.argv[1])
    root_dir = os.path.abspath(sys.argv[2])
    gui = bool(sys.argv[3])

    time.sleep(3)

    if _has_running_entrypoints():
        message = "Another instance of SmartQSM is currently running.\n\nPlease close all then retry."
        if gui:
            showerror("Update blocked", message)
        else:
            print(message)
        return 0

    try:
        copy_tree_overwrite(temp_dir, root_dir)
    except Exception:
        message = f"A serious error occurred during the upgrade process:\n\n{traceback.format_exc()}\n\nPlease make sure to upgrade after the next startup."
        if gui:
            root = tk.Tk()
            root.withdraw()
            showerror("Fatal error", message)
            root.destroy()
        else:
            print(message)
        try:
            os.remove(os.path.join(root_dir, "version.txt"))
        except Exception:
            pass
        sys.exit(1)

    try:
        shutil.rmtree(temp_dir, onexc=handle_remove_readonly)
    except Exception:
        pass

    post_installer_path = os.path.join(root_dir, "post_installation.py")
    if os.path.exists(post_installer_path):
        try:
            exec(open(post_installer_path, "r", encoding="utf-8").read())
        except Exception:
            message = f"{traceback.format_exc()}\nFailed to execute post_installation.py. Please try manually executing it."
            if gui:
                showerror("Error", message)
            else:
                print(message)
                exec("")

    version_file = os.path.join(root_dir, "version.txt")
    version = "0.0.0"
    try:
        with open(version_file, "r", encoding="utf-8") as f:
            version = f.read().strip()
    except Exception:
        pass

    message = f"SmartQSM has been successfully updated to version {version}!\n\nDo you want to know what has been updated?"
    if gui:
        root = tk.Tk()
        root.withdraw()
        if askyesno("Info", message):
            webbrowser.open("https://github.com/project-lightlin/SmartQSM")
        root.destroy()
    else:
        print(message)
        answer = input("[YES/ANY]").strip().lower()
        if answer == "yes":
            webbrowser.open("https://github.com/project-lightlin/SmartQSM")
        exec("")
    sys.exit(0)

if __name__ == "__main__":
    main()
        '''
        tmp_dir = tempfile.gettempdir()
        worker_path = os.path.join(tmp_dir, "smartqsm_update_worker.py")
        with open(worker_path, "w", encoding="utf-8") as wf:
            wf.write(worker_code)
        
        subprocess.Popen(
            [sys.executable, worker_path, temp_dir, ROOT_DIR, str(1 if gui else 0)],
            close_fds=True,
            stdout=sys.stdout,
            stderr=sys.stderr
        )

    except Exception:
        message = f"{traceback.format_exc()}\nFailed to start updater. Please try again later."
        if gui:
            showerror("Error", message)
        else:
            print(message)
        return 1

    return 1

def check_update(gui: bool = True) -> None:
    if _check_update(gui) == 1:
        sys.exit(0)
    if _root is not None:
        _root.destroy()