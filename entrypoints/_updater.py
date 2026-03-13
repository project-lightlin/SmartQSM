# SmartQSM - project-lightlin.github.io
# 
# Copyright (C) 2025-, YANG Jie <nj_yang_jie@foxmail.com>
# All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or 
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

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
import urllib
import json
import webview
import tempfile
from typing import Union
from bs4 import BeautifulSoup 

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

def show_changelog(current_version: Union[str, Version], json_source: str, gui: bool = True) -> bool:
    if type(current_version) == str:
        current_version = Version(current_version)

    json_data = None
    try:
        with urllib.request.urlopen(json_source) as response:
            json_data = json.loads(response.read().decode('utf-8'))
    except Exception:
        # For debugging: json_source is full content of a json file
        try:
            json_data = json.loads(json_source)
        except Exception:
            pass

    changelog_html = "<!DOCTYPE html><head><meta charset=\"UTF-8\"></head><body>"

    has_new_version = False
    if json_data is None:
        return False
    else:
        for release in json_data.get("releases", []):
            release_version = Version(release.get("version", "0.0.0"))
            if release_version > current_version:
                has_new_version = True
                version = release.get("version", "")
                codename = release.get("codename", "")
                released = release.get("released", "")
                changelog_html += f"<h2>Ver {version} {codename} <sup>released on {released}</sup></h2>"
                
                changelog_items = release.get("changelog", [])
                for idx, item in enumerate(changelog_items, 1):
                    label = item.get("label", "")
                    title = item.get("title", "")
                    
                    changelog_html += f"<h4>{idx}. <sup style=\"background-color: #C0C0C0;\">{label}</sup> {title}</h4>"
                    
                    desc_groups = item.get("description_group", [])
                    if desc_groups:
                        for desc in desc_groups:
                            if desc.get("media_type") == "text/html":
                                body = desc.get("body", "")
                                changelog_html += f"{body}"
                            elif desc.get("media_type") == "text/plain":
                                body = desc.get("body", "")
                                changelog_html += f"<p>{body}</p>"
                            else:
                                raise NotImplementedError(f"Unsupported media_type {desc.get("media_type")}")

    if not has_new_version:
        changelog_html += "<h2>The current version is the latest, no update is required.</h2>"

    changelog_html += "</body></html>"

    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html', encoding='utf-8')
    temp_file.write(changelog_html)
    temp_file.close()
    temp_html_path = temp_file.name
    if gui:
        try:
            webview.create_window(
                title="Changelog",
                url=temp_html_path,
                width=800,
                height=600,
                resizable=True
            )
            webview.start()
        finally:
            if temp_html_path and os.path.exists(temp_html_path):
                try:
                    os.unlink(temp_html_path)
                except:
                    pass
    else:
        soup = BeautifulSoup(changelog_html, "html.parser")
        print("========== Changelog ==========")
        plain_text = soup.get_text(strip=True, separator="\n")
        print(plain_text)
        print("===============================")
        input("Press ENTER to continue.")
    return True

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
    
    while not show_changelog(local_version, "https://raw.githubusercontent.com/project-lightlin/SmartQSM/refs/heads/main/changelog.json", gui):
        message = "Network communication is not smooth, do you want to retry to show the changelog? [YES/NO]"
        if gui:
            if not askyesno("Warning", message):
                break
        else:
            answer = input(message).strip().lower()
            while True:
                if answer == "yes":
                    break
                elif answer == "no":
                    break
            if answer == "no":
                break

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
        # Mainly used for configuring proxy
        git_config_path = os.path.join(ROOT_DIR, "entrypoints/git_config.txt")
        additional_arguments = []
        if os.path.exists(git_config_path):
            try:
                with open(git_config_path) as f:
                    git_config_lines = f.readlines()
                for line in git_config_lines:
                    argument = line.strip()
                    if argument != "":
                        if argument[0] != "#":
                            additional_arguments.append(argument)
            except Exception:
                pass

        cmd = ["git","clone"]
        for argument in additional_arguments:
            cmd += ["-c", argument]
        cmd += ["https://github.com/project-lightlin/SmartQSM.git", temp_dir]

        print("Command: "," ".join(cmd))

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
    message = "The update has been downloaded. Please confirm that there are no other running instances before upgrading! Please note that the prompt for success or failure will only appear after a period of time after the program terminates. Please be patient."
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
import importlib.util
import io

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
    gui = (sys.argv[3] == "1")

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
            spec = importlib.util.spec_from_file_location("post_installation", post_installer_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "run_post_install"):
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                try:
                    sys.stdout = io.StringIO()
                    sys.stderr = io.StringIO()
                    mod.run_post_install(root_dir)
                except Exception as post_err:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
                    raise post_err
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
        except Exception:
            message = f"{traceback.format_exc()}\n------------------------\nThe necessary features have been installed, but the post installation failed.\nPlease manually execute \"python.exe post_installation.py\" in the correct CONDA environment and under the correct working directory later."
            if gui:
                showerror("Error", message)
            else:
                print(message)
                exec("")
            sys.exit(1)

    version_file = os.path.join(root_dir, "version.txt")
    version = "0.0.0"
    try:
        with open(version_file, "r", encoding="utf-8") as f:
            version = f.read().strip()
    except Exception:
        pass

    message = f"SmartQSM has been successfully updated to version {version}!"
    if gui:
        root = tk.Tk()
        root.withdraw()
        showinfo("Info", message)
        root.destroy()
    else:
        print(message)
        input("Press ENTER to continue")
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
            stderr=sys.stderr,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        )
        
        os._exit(0)

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