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

import tkinter as tk
from tkinter.filedialog import askopenfilenames
from tkinter.messagebox import showerror, showinfo, showwarning
from tkinter import ttk
from _common_import import *
from typing import Dict, Optional, Any, List, Tuple
import argparse
import sys
import open3d.visualization.rendering as rendering
import traceback
import yaml
from utils.io3d import read_point_cloud
from datetime import datetime, timedelta
import time
import threading
import open3d as o3d
import logging
from logging.handlers import RotatingFileHandler
from core.skeletonization import Skeletonization
from core.refinement import Refinement
from core.modeling import Modeling
from core.parameterize_and_export import parameterize_and_export

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

TIP_ENABLE_VISUALIZER: str = "Display real-time results (Slow processing and high memory consumption)"
TIP_REPLACE_EXISTING_FILES: str = "Replace existing files."
TIP_ABOUT: str = "SmartQSM is a high-performance and easy-to-use method based on individual-tree point cloud data for 3D reconstruction and multi-scale parameter extraction.\nFor details, please visit project-lightlin.github.io\nLicense: AGPL-3.0"
TIP_RESULT_DESCRIPTION: str = "Result files will be saved in the same directory as the input files."
TIP_OVERRIDE_PROMPT: str = "When there is a configuration file with the same name as the point cloud file in the same directory, the configured value will be temporarily overwritten to apply to this point cloud file."

logging.getLogger("ezdxf").setLevel(logging.ERROR) 

logger = logging.getLogger("")
logger.setLevel(logging.INFO)
log_path = os.path.join(CUR_DIR, "smartqsm.log")
handler = RotatingFileHandler(
    filename=log_path,
    backupCount=10,
    encoding="utf-8"
)
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s",
    "%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)
logger.addHandler(handler)


class TkApp:
    _root: tk.Tk
    _file_list_variable: tk.Variable
    _file_list_box: tk.Listbox
    _file_overview_entry: tk.Entry
    _file_summary_string_var: tk.StringVar
    _config_name_to_file_path: Dict[str, str]
    _int_var_enable_visualizer: tk.IntVar
    _int_var_replace_existing_files: tk.IntVar
    _result: Optional[Dict[str, Any]] = None

    MESSAGE_NO_FILES: str = "No files."
    MESSAGE_ADDED_FILES: str = "Added {} files."
    MESSAGE_SELECTED_FILES: str = "Selected {} files."


    def __init__(self):
        self._config_name_to_file_path = {}
        self._load_configs()
        
        self._root = tk.Tk()
        self._root.title("SmartQSM (GUI mode) - Create project")
        self._root.geometry("640x720")
        
        tk.Button(self._root, text="About SmartQSM", command=lambda: showinfo(title="About", message=TIP_ABOUT)).pack(fill="x")
        
        ttk.Separator(self._root, orient='horizontal').pack(fill="x", pady=5)
        
        tk.Label(self._root, text="Import point cloud files:").pack(fill="x")
        tk.Button(self._root, text="Add files", command=self._open_files).pack(fill="x")

        file_list_frame = tk.Frame(self._root)
        file_list_frame.pack(fill="both", expand=True)

        file_list_box_vertical_scrollbar = tk.Scrollbar(file_list_frame, orient="vertical")
        file_list_box_vertical_scrollbar.pack(side="right", fill="y")

        self._file_list_variable = tk.Variable(value=list())
        self._file_list_variable.trace("w", self._update_file_summary)
        self._file_list_box = tk.Listbox(file_list_frame, listvariable=self._file_list_variable,selectmode=tk.MULTIPLE,  yscrollcommand=file_list_box_vertical_scrollbar.set)
        self._file_list_box.pack(fill="both", expand=True)
        self._file_list_box.bind('<<ListboxSelect>>', self._update_file_summary)
        file_list_box_vertical_scrollbar.config(command=self._file_list_box.yview)
        
        self._file_summary_string_var = tk.StringVar(value=TkApp.MESSAGE_NO_FILES)
        self._file_overview_entry = tk.Entry(self._root, textvariable=self._file_summary_string_var, state="readonly")
        self._file_overview_entry.pack(fill="x")    

        tk.Button(self._root, text="Remove selected files", command=self._remove_selected_files).pack(fill="x")
        tk.Button(self._root, text="Clear list", command=self._clear_file_list).pack(fill="x")

        ttk.Separator(self._root, orient='horizontal').pack(fill="x", pady=5)

        tk.Label(self._root, text="Select a configuration file:").pack(fill="x")
        
        text = tk.Text(self._root, height=3)
        text.insert(tk.END, TIP_OVERRIDE_PROMPT)
        text.config(state="disabled")
        text.pack(fill="x")

        config_frame = tk.Frame(self._root)
        config_frame.pack(fill="x")

        config_list_box_vertical_scrollbar = tk.Scrollbar(config_frame, orient="vertical")
        config_list_box_vertical_scrollbar.pack(side="right", fill="y")

        self._config_list_variable = tk.Variable(value=list(self._config_name_to_file_path.keys()))
        self._config_list_box = tk.Listbox(config_frame, listvariable=self._config_list_variable, height=6, selectmode=tk.SINGLE,  yscrollcommand=config_list_box_vertical_scrollbar.set)
        self._config_list_box.pack(fill="x")
        config_list_box_vertical_scrollbar.config(command=self._config_list_box.yview)

        ttk.Separator(self._root, orient='horizontal').pack(fill="x", pady=5)

        tk.Label(self._root, text=TIP_RESULT_DESCRIPTION).pack(fill="x")
        self._int_var_enable_visualizer = tk.IntVar(value=0)
        tk.Checkbutton(self._root, text=TIP_ENABLE_VISUALIZER, variable=self._int_var_enable_visualizer).pack(anchor="w")
        self._int_var_replace_existing_files = tk.IntVar(value=0)
        tk.Checkbutton(self._root, text=TIP_REPLACE_EXISTING_FILES, variable=self._int_var_replace_existing_files).pack(anchor="w")

        tk.Button(self._root, text="Process", command=self._run).pack(fill="x")

        # Initialize
        self._result = None

        self._root.mainloop()
        return
    
    def _load_configs(self):
        try:
            config_dir = os.path.join(ROOT_DIR, "configs")
            for basename in os.listdir(config_dir):
                stem, extension = os.path.splitext(basename)
                if extension in [".yaml", ".yml"]:
                    self._config_name_to_file_path[stem] = os.path.join(config_dir, basename)
            if len(self._config_name_to_file_path) == 0:
                raise ValueError
        except Exception:
            showerror("Error", "No configuration files found. Please place at least one configuration file in directory 'configs'.")
            sys.exit(2)

    def _open_files(self):
        paths = askopenfilenames(
            title="Open",
            filetypes=[
                ('LAS file', '*.las *.laz'), 
                ('ASCII point cloud', '*.txt *.xyz *.xyzn *.xyzrgb *.pts'), 
                ("PLY point cloud", ".ply"),
                ("Point Cloud Library cloud", ".pcd"),
                ('All files', '*')
            ]
        )
        if not paths:
            return
        
        opened_paths = list(self._file_list_variable.get())
        for path in paths:
            if path in opened_paths:
                showwarning(title="Warning", message="File {} is already opened.".format(path))
                continue
            opened_paths.append(path)
        self._file_list_variable.set(opened_paths)
    
    def _remove_selected_files(self):
        selected_indices = self._file_list_box.curselection()
        if selected_indices:
            file_list = list(self._file_list_variable.get())
            for idx in sorted(selected_indices, reverse=True):
                if 0 <= idx < len(file_list):
                    del file_list[idx]
            self._file_list_variable.set(file_list)
            self._file_list_box.selection_clear(0, tk.END)

    def _clear_file_list(self):
        self._file_list_variable.set(())

    def _update_file_summary(self, *args):
        num_files = len(self._file_list_variable.get())
        file_summary: str
        if num_files == 0:
            file_summary = self.MESSAGE_NO_FILES
        else:
            file_summary = self.MESSAGE_ADDED_FILES.format(num_files)
        num_selected_files = len(self._file_list_box.curselection())
        if num_selected_files > 0:
            file_summary += " " + self.MESSAGE_SELECTED_FILES.format(num_selected_files)
        self._file_summary_string_var.set(file_summary)
        
    def _run(self):
        file_list = self._file_list_variable.get()
        if len(file_list) == 0:
            showerror("Error", "No files selected. Please select at least one file.")
            return
        config_list_idx = self._config_list_box.curselection()
        if not config_list_idx:
            showerror("Error", "No configuration file selected. Please select at least one configuration file.")
            return
        config_name = self._config_list_box.get(config_list_idx[0])
        self._result = {
            "files": file_list,
            "config": self._config_name_to_file_path[config_name],
            "enable_visualizer": bool(self._int_var_enable_visualizer.get()),
            "replace_existing_files": bool(self._int_var_replace_existing_files.get()),
            "run_in_terminal": False
        }
        self._root.destroy()
    
    def fetch_result(self):
        return self._result
    
class Open3DApp:
    _window: gui.Window
    _vert: gui.Vert
    _scene_widget: Optional[gui.SceneWidget]
    _total_progress_bar: gui.ProgressBar
    _total_title_label: gui.Label
    _total_status_label: gui.Label
    _current_title_label: gui.Label
    _current_progress_bar: gui.ProgressBar
    _current_status_label: gui.Label
    _processed_file_collapsable_vert: gui.CollapsableVert
    _processed_file_tree_view: gui.TreeView
    _failed_file_collapsable_vert: gui.CollapsableVert
    _failed_file_tree_view: gui.TreeView
    _skipped_file_collapsable_vert: gui.CollapsableVert
    _skipped_file_tree_view: gui.TreeView
    _config: Dict[str, Any]
    _can_existing_files_be_replaced: bool

    TITLE_TOTAL: str = "Total: "
    TITLE_CURRENT: str = "Current: "
    TITLE_PROCESSED_FILES: str = "Processed files: "
    TITLE_FAILED_FILES: str = "Failed files: "
    TITLE_SKIPPED_FILES: str = "Skipped files: "
    WINDOW_TITLE_RECONSTUCTING: str = "SmartQSM is reconstructing trees..."
    WINDOW_TITLE_FINISHED: str = "SmartQSM - Reconstruction completed"
    
    _processed_files: List[str]
    _failed_files: List[str]
    _skipped_files: List[str]

    _path_to_file_size: Dict[str, int]

    _processed_byte_length: int
    _start_datetime: Optional[datetime]
    _due_datetime: Optional[datetime]
    _current_info: str
    _current_status: str
    _current_progress: float
    _new_processed_file_start_idx: int
    _new_failed_file_start_idx: int
    _new_skipped_file_start_idx: int

    _thread1: Optional[threading.Thread] 
    _thread2: Optional[threading.Thread]
    _terminated: bool

    _dialog_setting: Optional[Dict[str, Any]]

    _config_path: str

    def __init__(self, config):
        is_visualizer_enabled: bool = config["enable_visualizer"]
        self._can_existing_files_be_replaced: bool = config["replace_existing_files"]
        self._processed_files = []
        self._failed_files = []
        self._skipped_files = []
        self._path_to_file_size = {}
        self._current_info = ""
        self._current_status = ""
        self._current_progress = 0.0
        self._new_processed_file_start_idx = 0
        self._new_failed_file_start_idx = 0
        self._new_skipped_file_start_idx = 0
        self._processed_byte_length = 0
        self._start_datetime = None
        self._due_datetime = None
        self._thread1 = None
        self._thread2 = None
        self._terminated = False
        
        for file_path in config["files"]:
            self._path_to_file_size[file_path] = os.path.getsize(file_path)

        # UI
        self._window = gui.Application.instance.create_window(Open3DApp.WINDOW_TITLE_RECONSTUCTING, width=1280 if is_visualizer_enabled else 640, height = 720)
        em = self._window.theme.font_size

        self._vert = gui.Vert(em, gui.Margins(em, em, em, em))
        
        row1 = gui.Vert(0, gui.Margins(0, 0, 0, 0))
        self._total_title_label = gui.Label(Open3DApp.TITLE_TOTAL)
        row1.add_child(self._total_title_label)
        self._total_progress_bar = gui.ProgressBar()
        row1.add_child(self._total_progress_bar)
        self._total_status_label = gui.Label("")
        row1.add_child(self._total_status_label)
        self._vert.add_child(row1)
        
        row2 = gui.Vert(0, gui.Margins(0, 0, 0, 0))
        self._current_title_label = gui.Label(Open3DApp.TITLE_CURRENT)
        row2.add_child(self._current_title_label)
        self._current_progress_bar = gui.ProgressBar()
        row2.add_child(self._current_progress_bar)
        self._current_status_label = gui.Label("")
        row2.add_child(self._current_status_label)
        self._vert.add_child(row2)
        
        self._processed_file_collapsable_vert = gui.CollapsableVert(Open3DApp.TITLE_PROCESSED_FILES)
        self._processed_file_tree_view = gui.TreeView()
        self._processed_file_collapsable_vert.add_child(self._processed_file_tree_view)
        self._vert.add_child(self._processed_file_collapsable_vert)

        self._failed_file_collapsable_vert = gui.CollapsableVert(Open3DApp.TITLE_FAILED_FILES)
        self._failed_file_tree_view = gui.TreeView()
        self._failed_file_collapsable_vert.add_child(self._failed_file_tree_view)
        self._vert.add_child(self._failed_file_collapsable_vert)

        self._skipped_file_collapsable_vert = gui.CollapsableVert(Open3DApp.TITLE_SKIPPED_FILES)
        self._skipped_file_tree_view = gui.TreeView()
        self._skipped_file_collapsable_vert.add_child(self._skipped_file_tree_view)
        self._vert.add_child(self._skipped_file_collapsable_vert)

        self._scene_widget = None
        if is_visualizer_enabled:
            self._scene_widget = gui.SceneWidget()
            self._scene_widget.scene = rendering.Open3DScene(self._window.renderer)
            self._scene_widget.scene.set_lighting(self._scene_widget.scene.LightingProfile.MED_SHADOWS, np.array([0., 0., -1.]))
            self._window.add_child(self._scene_widget)

        self._window.add_child(self._vert)
        self._window.set_on_layout(self._on_layout)
        self._window.set_on_close(self._on_close)

        self._dialog_setting = None

        # Initialize
        self._failed_file_collapsable_vert.set_is_open(True)
        self._skipped_file_collapsable_vert.set_is_open(True)
        
        self._config = {}
        try:
            self._config_path = config["config"]
            self._config = Open3DApp.load_or_update_smartqsm_config(config["config"], self._config)
            logger.info(f"Loaded standard config file.")
        except Exception:
            self._terminated = True
            error_info = "Standard config file {} has been illegally modified or damaged.".format(config["config"])
            logger.error(error_info)
            self._window.show_message_box(
                "", # Title does not display
                error_info
            )
        return

    def _on_layout(self, layout_context):
        content_rect = self._window.content_rect

        if self._scene_widget is None:
            self._vert.frame = gui.Rect(
                content_rect.x, 
                content_rect.y, 
                content_rect.width,  
                content_rect.height
            )
        else:
            half_width = content_rect.width // 2
            height = content_rect.height

            self._scene_widget.frame = gui.Rect(
                content_rect.x,
                content_rect.y,
                half_width,
                height
            )
            
            self._vert.frame = gui.Rect(
                content_rect.x + half_width, 
                content_rect.y, 
                half_width,
                height
            )
        
        if self._dialog_setting is not None: 
            # When a window is minimized and the show_dialogmethod (including show_message_box) is called, 
            # the dialog's position and size are incorrect. 
            # After modifying the dimensions (and the dimensions of the child controls) in on_layout, 
            # only the dialog's frame is displayed, while the content is not displayed (even if visible = True).

            def builtin_fn():
                gui.Application.instance.post_to_main_thread(self._window, self._window.close_dialog)
                gui.Application.instance.post_to_main_thread(self._window, lambda:
                    show_multibutton_messagebox(**self._dialog_setting)
                )
            threading.Thread(target=builtin_fn, daemon=True).start()
        return
    
    def _on_close_confirm(self):
        self._window.close_dialog()
        message = "Terminating... The program will automatically close after cleaning is complete. \nPlease do not forcefully abort."
        show_snackbar(self._window, message)
        print(message)
        if self._thread1 is not None and self._thread1.is_alive():
            self._thread1._stop()
        if self._thread2 is not None and self._thread2.is_alive():
            self._thread2._stop()
        
        gui.Application.instance.quit()
        return

    
    def _on_close(self):
        if not self._terminated:
            show_multibutton_messagebox(
                self._window,
                "Warning",
                "The program is currently executing.\nAre you sure you want to terminate and close the program?",
                {
                    "Yes": self._on_close_confirm,
                    "No": self._window.close_dialog
                }
            )
        else:
            gui.Application.instance.quit()
        return
    
    def _refresh_gui_during_process(self):
        elapsed_second = (datetime.now() - self._start_datetime).total_seconds()
        hours = int(elapsed_second // 3600)  
        elapsed_seconds_after_hours = elapsed_second % 3600 
        minutes = int(elapsed_seconds_after_hours // 60)  
        seconds = int(elapsed_seconds_after_hours % 60)  
        elapsed_time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        remaining_time_str = "--:--:--"
        if self._due_datetime is not None:
            remaining_second = max(0, (self._due_datetime - datetime.now()).total_seconds())
            hours = int(remaining_second // 3600)
            remaining_seconds_after_hours = remaining_second % 3600 
            minutes = int(remaining_seconds_after_hours // 60)  
            seconds = int(remaining_seconds_after_hours % 60)  
            if hours > 99:
                hours = 99
                minutes = 59
                seconds = 59
            remaining_time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        # Prevent changes
        processed_files = self._processed_files.copy()
        failed_files = self._failed_files.copy()
        skipped_files = self._skipped_files.copy()

        num_passed_files = len(processed_files) + len(failed_files) + len(skipped_files)
        overall_completion_rate = num_passed_files / len(self._path_to_file_size)
        self._total_title_label.text = Open3DApp.TITLE_TOTAL + f"{len(self._path_to_file_size)}; Passed {num_passed_files} (Failed {len(failed_files)} / Skipped {len(skipped_files)})"
        self._window.title = f"({overall_completion_rate * 100: .2f}%, remaining: {remaining_time_str}) " + Open3DApp.WINDOW_TITLE_RECONSTUCTING
        self._total_progress_bar.value = overall_completion_rate
        self._total_status_label.text = f"({overall_completion_rate * 100: .2f}%) Elapsed: {elapsed_time_str} / Remaining: {remaining_time_str}"
        self._current_title_label.text = Open3DApp.TITLE_CURRENT + f"{self._current_info}"
        self._refresh_scene(with_progress=True)
        
        for i in range(self._new_processed_file_start_idx, len(self._processed_files)):
            self._processed_file_tree_view.add_text_item(self._processed_file_tree_view.get_root_item(), self._processed_files[i])
        self._new_processed_file_start_idx = len(self._processed_files)
        for i in range(self._new_failed_file_start_idx, len(self._failed_files)):
            self._failed_file_tree_view.add_text_item(self._failed_file_tree_view.get_root_item(), self._failed_files[i])
        self._new_failed_file_start_idx = len(self._failed_files)
        for i in range(self._new_skipped_file_start_idx, len(self._skipped_files)):
            self._skipped_file_tree_view.add_text_item(self._skipped_file_tree_view.get_root_item(), self._skipped_files[i])
        self._new_skipped_file_start_idx = len(self._skipped_files)
        return

    def _refresh_scene(self, *args, with_progress: bool = True):
        if len(args) != 0 and self._scene_widget is not None:
            self._scene_widget.scene.clear_geometry()
            material_record = rendering.MaterialRecord()
            material_record.shader = "defaultLit"
            for i, arg in enumerate(args):
                if type(arg) == o3d.geometry.PointCloud:
                    self._scene_widget.scene.add_geometry(str(i), arg, material_record)
                elif type(arg) == o3d.geometry.TriangleMesh:
                    self._scene_widget.scene.add_geometry(str(i), arg, material_record)
                elif type(arg) == o3d.geometry.LineSet:
                    self._scene_widget.scene.add_geometry(str(i), arg, material_record)
                elif type(arg) == str:
                    self._current_status = arg
            if not self._scene_widget.scene.bounding_box.is_empty():
                bounds = self._scene_widget.scene.bounding_box
                self._scene_widget.setup_camera(60, bounds, bounds.get_center())
                # front view
                cam = self._scene_widget.scene.camera
                view_mat = cam.get_view_matrix()
                inv_view_mat = np.linalg.inv(view_mat)
                eye = inv_view_mat[:3, 3]
                center_of_rotation = self._scene_widget.center_of_rotation
                distance = np.linalg.norm(eye - center_of_rotation)
                self._scene_widget.look_at(center_of_rotation, center_of_rotation - np.array([0, 1, 0]) * distance, np.array([0, 0, 1]))
        
        self._current_progress_bar.value = self._current_progress
        self._current_status_label.text = f"({self._current_progress * 100:.2f}%) {self._current_status}" if with_progress else self._current_status
        self._window.post_redraw()
        return

    def _close_dialog_and_clear_dialog_setting(self):
        if self._dialog_setting is not None:
            self._dialog_setting["window"].close_dialog()
            self._dialog_setting = None

    def _refresh_gui_after_process(self):
        self._current_info = ""
        self._current_progress = 0.
        self._current_status = ""
        self._refresh_gui_during_process()
        self._refresh_scene(with_progress=False)
        self._window.title = Open3DApp.WINDOW_TITLE_FINISHED
        
        message = "Reconstruction completed. " if len(self._failed_files) == 0 else f"Partial reconstruction completed. "
        if len(self._failed_files) > 0:
            message += f"\n - Failed {len(self._failed_files)} files. Check log file {log_path} for more details."
        if len(self._skipped_files) > 0:
            message += f"\n - Skipped {len(self._skipped_files)} files. There may be non point cloud files (marked with prefix [X]), please check."
        message +"\n\n\n"

        self._dialog_setting = {
            "window": self._window, 
            "title": "", 
            "message": message, 
            "button_text_to_callback": {
                "OK": self._close_dialog_and_clear_dialog_setting
            }
        }
        show_multibutton_messagebox(**self._dialog_setting)
        return

    @staticmethod
    def load_or_update_smartqsm_config(new_config_path: str, current_config: Optional[Dict[str, Any]] = None):
        new_config: Dict[str, Any] = current_config.copy() if current_config else {}
        with open(new_config_path, "r") as f:
            config: Dict[str, Any] = yaml.safe_load(f)
            new_config.update(config)
        # Check
        new_config["skeletonization"]
        new_config["refinement"]
        new_config["modeling"]
        return new_config

    def start(self):
        if len(self._config) == 0:
            return
        self._start_datetime = datetime.now()

        def continuously_update_gui():
            while True:
                if len(self._processed_files) + len(self._failed_files) + len(self._skipped_files) == len(self._path_to_file_size):
                    break
                gui.Application.instance.post_to_main_thread(self._window, self._refresh_gui_during_process)
                time.sleep(1.)
            gui.Application.instance.post_to_main_thread(self._window, self._refresh_gui_after_process)
        self._thread1 = threading.Thread(target=continuously_update_gui, daemon=True).start()

        self._thread2 = threading.Thread(target=self._run_smartqsm, daemon=True).start()
    
    def _run_smartqsm(self):
        total_file_size = sum(self._path_to_file_size.values())
        passed_file_size = 0
        
        for i, (cloud_path, file_size) in enumerate(self._path_to_file_size.items()):
            self._current_info = cloud_path
            gui.Application.instance.post_to_main_thread(self._window, self._refresh_gui_during_process) # Refresh to update current file

            self._current_progress = 0.
            self._current_status = "Preparing..."
            gui.Application.instance.post_to_main_thread(self._window, lambda: self._refresh_scene())

            logger.info(f"({i + 1}/{len(self._path_to_file_size)}) Started processing {cloud_path} ({file_size / 1024 ** 2:.2f} MB)")

            config = self._config.copy()
            prefix = ""
            try:
                path_without_extension, extension = os.path.splitext(cloud_path)
                if os.path.exists(path_without_extension + "_qsm.mat") and self._can_existing_files_be_replaced == False:
                    raise FileExistsError
                is_custom_config_existed = False
                for extension in [".yml", ".yaml"]:
                    custom_config_path = path_without_extension + extension

                    if os.path.exists(custom_config_path):
                        is_custom_config_existed = True
                        try:
                            config = Open3DApp.load_or_update_smartqsm_config(custom_config_path, config)
                            prefix += "[@] "
                            logger.info(f"Loaded external config file {custom_config_path} to update config {self._config_path}.")
                            self._current_info = f"{cloud_path} (*Apply config ./{os.path.basename(custom_config_path)})"
                            break
                        except Exception:
                            prefix += "[!]  "
                            logger.warning(f"Skipped invalid custom config file {custom_config_path}\n"+traceback.format_exc())
                            logger.warning(f"Restore to {self._config_path}")
                            self._current_info = f"{cloud_path} (!Bad custom config. Restore to {os.path.basename(self._config_path)})"
                            break
                if is_custom_config_existed == False:
                    logger.info(f"Use default config: {self._config_path}")
                    self._current_info = f"{cloud_path} (Default config: {os.path.basename(self._config_path)})"
                    
                cloud = None
                projection = None

                try:
                    cloud, projection = read_point_cloud(cloud_path)
                    if len(cloud.points) == 0:
                        raise ValueError
                except Exception:
                    prefix += "[X]  "
                    logger.warning(f"{cloud_path} is not an valid point cloud.")
                    raise FileExistsError
                
                aabb = cloud.get_axis_aligned_bounding_box()
                global_shift = np.array([
                    -(aabb.min_bound[0] + aabb.max_bound[0]) / 2.0,
                    -(aabb.min_bound[1] + aabb.max_bound[1]) / 2.0,
                    -aabb.min_bound[2]
                ], dtype = np.float64)
                cloud.translate(global_shift)
                cloud = cloud.voxel_down_sample(voxel_size=config.get("voxel_down_size", 0.01))

                filter: str = config.get("filter", None)
                if filter is None:
                    pass
                elif filter == "statistical_outlier_removal":
                    cloud = cloud.remove_statistical_outlier(
                        **config["params_for_filter"],
                    )[0]
                elif filter == "radius_outlier_removal":
                    cloud = cloud.remove_radius_outlier(
                        **config["params_for_filter"],
                    )[0]
                else:
                    raise NotImplementedError(f"Filter {filter} is not implemented.")

                gui.Application.instance.post_to_main_thread(self._window, lambda: self._refresh_scene("Initialized point cloud. Start reconstructing.", cloud))

                points = np.asarray(cloud.points)

                skeletonization = Skeletonization(
                    verbose=self._scene_widget is not None
                )
                skeletonization.set_params(**config["skeletonization"])
                current_step = 0
                total_step = len(skeletonization)
                generator = skeletonization.run(points=points)
                
                self._current_status = "Extracting skeleton..."
                gui.Application.instance.post_to_main_thread(self._window, lambda: self._refresh_scene())

                skeletal_points = None
                skeleton = None
                radii = None

                while True:
                    self._current_progress = 0. + current_step / total_step / 4.0
                    current_step += 1
                    try:
                        result = next(generator)
                        if not isinstance(result, tuple):
                            result = (result,)

                        gui.Application.instance.post_to_main_thread(self._window, lambda: self._refresh_scene(*result))
                    except StopIteration as e:
                        self._current_progress = 0.25
                        self._current_status = "Waiting..."
                        gui.Application.instance.post_to_main_thread(self._window, lambda: self._refresh_scene())

                        skeletal_points, skeleton, radii = e.value
                        break

                refinement = Refinement(
                    verbose=self._scene_widget is not None
                )
                refinement.set_params(**config["refinement"])
                current_step = 0
                total_step = len(refinement)
                generator = refinement.run(points, skeletal_points, skeleton, radii)

                self._current_status = "Refining skeleton..."
                gui.Application.instance.post_to_main_thread(self._window, lambda: self._refresh_scene())

                branch_id_to_branch = None

                while True:
                    self._current_progress = 0.25 + current_step / total_step / 4.0
                    current_step += 1
                    try:
                        result = next(generator)
                        if not isinstance(result, tuple):
                            result = (result,)

                        gui.Application.instance.post_to_main_thread(self._window, lambda: self._refresh_scene(*result))
                    except StopIteration as e:
                        self._current_progress = 0.5
                        self._current_status = "Waiting..."
                        gui.Application.instance.post_to_main_thread(self._window, lambda: self._refresh_scene())

                        branch_id_to_branch = e.value
                        break
                
                modeling = Modeling(
                    verbose=self._scene_widget is not None,
                )
                modeling.set_params(**config["modeling"])
                current_step = 0
                total_step = len(modeling)
                generator = modeling.run(branch_id_to_branch)

                self._current_status = "Modeling..."
                gui.Application.instance.post_to_main_thread(self._window, lambda: self._refresh_scene())

                while True:
                    self._current_progress = 0.5 + current_step / total_step / 4.0
                    current_step += 1
                    try:
                        result = next(generator)
                        if not isinstance(result, tuple):
                            result = (result,)

                        gui.Application.instance.post_to_main_thread(self._window, lambda: self._refresh_scene(*result))
                    except StopIteration as e:
                        self._current_progress = 0.75
                        self._current_status = "Waiting..."
                        gui.Application.instance.post_to_main_thread(self._window, lambda: self._refresh_scene())

                        branch_id_to_branch = e.value
                        break
                
                self._current_status = "Parameterizing and exporting..."
                gui.Application.instance.post_to_main_thread(self._window, lambda: self._refresh_scene())

                parameterize_and_export(
                    branch_id_to_branch,
                    points,
                    path_without_extension,
                    global_shift,
                    projection
                )

                self._current_progress = 1.
                self._current_status = "Done."
                gui.Application.instance.post_to_main_thread(self._window, lambda: self._refresh_scene())

                logger.info(f"Done.")
                self._processed_files.append(prefix + cloud_path)
            except FileExistsError:
                self._current_progress = 1.
                self._current_status = "Skipped."
                gui.Application.instance.post_to_main_thread(self._window, lambda: self._refresh_scene())

                logger.warning(f"Skipped.")
                self._skipped_files.append(prefix + cloud_path)
            except Exception:
                self._current_progress = 1.
                self._current_status = "Failed."
                gui.Application.instance.post_to_main_thread(self._window, lambda: self._refresh_scene())

                logger.error(traceback.format_exc())
                logger.info(f"Failed.")
                self._failed_files.append(prefix + cloud_path)

            passed_file_size += file_size
            elapsed_time = (datetime.now() - self._start_datetime).total_seconds()
            speed = passed_file_size / (elapsed_time + 1) # Prevent division by 0
            self._due_datetime = self._start_datetime + timedelta(seconds= total_file_size / (speed + 1)) # Prevent division by 0
        self._terminated = True

def process_in_terminal(
    **kwargs
):
    can_existing_files_be_replaced = kwargs["replace_existing_files"]
    config_path = kwargs["config"]

    path_to_file_size = {}
    for file_path in kwargs["files"]:
        path_to_file_size[file_path] = os.path.getsize(file_path)

    base_config = Open3DApp.load_or_update_smartqsm_config(config_path)
    processed_files = []
    skipped_files = []
    failed_files = []
    
    for i, (cloud_path, file_size) in enumerate(path_to_file_size.items()):
        output_text = f"({i + 1}/{len(path_to_file_size)}) Started processing {cloud_path} ({file_size / 1024 ** 2:.2f} MB)"
        logger.info(output_text)
        print(output_text)
        config = base_config.copy()
        prefix = ""
        try:
            path_without_extension, extension = os.path.splitext(cloud_path)
            if os.path.exists(path_without_extension + "_qsm.mat") and can_existing_files_be_replaced == False:
                raise FileExistsError
            is_custom_config_existed = False
            for extension in [".yml", ".yaml"]:
                custom_config_path = path_without_extension + extension

                if os.path.exists(custom_config_path):
                    is_custom_config_existed = True
                    try:
                        config = Open3DApp.load_or_update_smartqsm_config(custom_config_path, config)
                        prefix += "[@] "
                        logger.info(f"Loaded external config file {custom_config_path} to update config {config_path}.")
                        current_info = f"{cloud_path} (*Apply config ./{os.path.basename(custom_config_path)})"
                        break
                    except Exception:
                        prefix += "[!]  "
                        logger.warning(f"Skipped invalid custom config file {custom_config_path}\n"+traceback.format_exc())
                        logger.warning(f"Restore to {config_path}")
                        current_info = f"{cloud_path} (!Bad custom config. Restore to {os.path.basename(config_path)})"
                        break
            if is_custom_config_existed == False:
                logger.info(f"Use default config: {config_path}")
                current_info = f"{cloud_path} (Default config: {os.path.basename(config_path)})"
            print(current_info)

            cloud = None
            projection = None

            try:
                cloud, projection = read_point_cloud(cloud_path)
                if len(cloud.points) == 0:
                    raise ValueError
            except Exception:
                prefix += "[X]  "
                logger.warning(f"{cloud_path} is not an valid point cloud.")
                raise FileExistsError
            
            aabb = cloud.get_axis_aligned_bounding_box()
            global_shift = np.array([
                -(aabb.min_bound[0] + aabb.max_bound[0]) / 2.0,
                -(aabb.min_bound[1] + aabb.max_bound[1]) / 2.0,
                -aabb.min_bound[2]
            ], dtype = np.float64)
            cloud.translate(global_shift)
            cloud = cloud.voxel_down_sample(voxel_size=config.get("voxel_down_size", 0.01))

            filter: str = config.get("filter", None)
            if filter is None:
                pass
            elif filter == "statistical_outlier_removal":
                cloud = cloud.remove_statistical_outlier(
                    **config["params_for_filter"],
                )[0]
            elif filter == "radius_outlier_removal":
                cloud = cloud.remove_radius_outlier(
                    **config["params_for_filter"],
                )[0]
            else:
                raise NotImplementedError(f"Filter {filter} is not implemented.")

            print("Initialized point cloud. Start reconstructing.")

            points = np.asarray(cloud.points)

            skeletonization = Skeletonization(
                verbose=False
            )
            skeletonization.set_params(**config["skeletonization"])
            generator = skeletonization.run(points=points)
            
            print("Extracting skeleton...")

            skeletal_points = None
            skeleton = None
            radii = None

            while True:
                try:
                    result = next(generator)
                    if not isinstance(result, tuple):
                        result = (result,)

                except StopIteration as e:
                    print("Waiting...")

                    skeletal_points, skeleton, radii = e.value
                    break

            refinement = Refinement(
                verbose=False
            )
            refinement.set_params(**config["refinement"])
            generator = refinement.run(points, skeletal_points, skeleton, radii)

            print("Refining skeleton...")

            branch_id_to_branch = None

            while True:
                try:
                    result = next(generator)
                    if not isinstance(result, tuple):
                        result = (result,)

                except StopIteration as e:
                    print("Waiting...")

                    branch_id_to_branch = e.value
                    break
            
            modeling = Modeling(
                verbose=False,
            )
            modeling.set_params(**config["modeling"])
            generator = modeling.run(branch_id_to_branch)

            print("Modeling...")

            while True:
                try:
                    result = next(generator)
                    if not isinstance(result, tuple):
                        result = (result,)

                except StopIteration as e:
                    print("Waiting...")

                    branch_id_to_branch = e.value
                    break
            
            print("Parameterizing and exporting...")

            parameterize_and_export(
                branch_id_to_branch,
                points,
                path_without_extension,
                global_shift,
                projection
            )

            print("Done.")

            logger.info(f"Done.")
            processed_files.append(prefix + cloud_path)
        except FileExistsError:
            print("Skipped.")

            logger.warning(f"Skipped.")
            skipped_files.append(prefix + cloud_path)
        except Exception:
            print("Failed.")

            logger.error(traceback.format_exc())
            logger.info(f"Failed.")
            failed_files.append(prefix + cloud_path)

    print("Finished.")
    if skipped_files:
        print(f"Skipped files: \n{"\n".join(skipped_files)}")
        print("There may be non point cloud files (marked with prefix [X]), please check.")
    if failed_files:
        print(f"Failed files: \n{"\n".join(failed_files)}")
        print(f"Check log file {log_path} for more details.")
    

def main():
    argument_parser = argparse.ArgumentParser(
        prog="python smartqsm.py",
        description="SmartQSM (CLI mode)",
        epilog="Start GUI mode without any parameters.\n\n"+ TIP_RESULT_DESCRIPTION + "\n\n" + TIP_ABOUT,
        formatter_class=argparse.RawTextHelpFormatter
    )
    argument_parser.add_argument("-v", action="store_true", help=TIP_ENABLE_VISUALIZER)
    argument_parser.add_argument("-y", action="store_true", help=TIP_REPLACE_EXISTING_FILES)
    argument_parser.add_argument("-c", "--config", help=f"Standard configuration file. ({TIP_OVERRIDE_PROMPT})")
    argument_parser.add_argument("-t", "--terminal", action="store_true", help="Run in terminal mode. After activation, '-v' automatically becomes false.")
    argument_parser.add_argument("CLOUD_PATHS", nargs="*", help="Individual-tree point cloud file(s)")

    logger.info("Program started.")

    if len(sys.argv) > 1:
        args, unknown = argument_parser.parse_known_args()
        try:
            if unknown:
                print("Unknown argument(s):", " ".join(unknown))
                raise Exception
            if not args.config:
                print("No configuration file specified.")
                raise Exception
            if not str(args.config).endswith((".yaml", ".yml")):
                print("Invalid configuration file {}.".format(args.config))
                raise Exception
            if not os.path.exists(args.config):
                print("Configuration file {} does not exist.".format(args.config))
                raise Exception
            if not args.CLOUD_PATHS:
                print("No point cloud file specified.")
                raise Exception
            result = {
                "files": args.CLOUD_PATHS,
                "config": args.config,
                "enable_visualizer": args.v,
                "replace_existing_files": args.y,
                "run_in_terminal": args.terminal
            }
            if result["run_in_terminal"]:
                result["enable_visualizer"] = False
        except Exception:
            argument_parser.print_help()
            sys.exit(2)
    else:
        tk_app = TkApp()
        result = tk_app.fetch_result()
        if result is None:
            exit(0)
    
    result["files"] = [os.path.abspath(f) for f in result["files"]]
    result["config"] = os.path.abspath(result["config"])

    message = f"Project started with params: {str(result)}"
    logger.info(message)
    if result["run_in_terminal"]:
        process_in_terminal(**result)
    else:
        gui.Application.instance.initialize()
        set_default_font_description()
        open3d_app = Open3DApp(result)
        open3d_app.start()
        gui.Application.instance.run()


if __name__ == "__main__":
    from filelock import FileLock, Timeout

    LOCK_FILE = os.path.join(ROOT_DIR, "entrypoints/smartqsm.run")
    
    update_check_is_required = True
    if os.path.exists(LOCK_FILE):
        try:
            os.remove(LOCK_FILE)
        except Exception:
            update_check_is_required = False
    if update_check_is_required:
        if len(sys.argv) > 1:
            check_update(False)
        else:
            check_update()

    try: 
        lock = FileLock(LOCK_FILE, timeout=0)
        lock.acquire()
    except Timeout:
        message = "Another instance of SmartQSM is running."
        if len(sys.argv) > 1:
            print(message)
        else:
            from tkinter.messagebox import showwarning
            showwarning("Warning", message)
        exit(1)
        
    try:
        handler.doRollover()
        main()
    finally:
        lock.release()