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
from tkinter.filedialog import askopenfilenames, asksaveasfilename
from tkinter.messagebox import showerror, showinfo, showwarning
from tkinter import ttk
import os
import yaml
import pandas as pd
from scipy.io import loadmat
import numpy as np
from pathlib import Path
import xlsxwriter
from _common_import import *

APP_DIR = os.path.dirname(os.path.abspath(__file__))

TIP_ABOUT: str = "Parameter exporter is a part of the SmartQSM toolkit.\nFor details, please visit project-lightlin.github.io\nLicense: AGPL-3.0"

class App:
    MESSAGE_NO_FILES: str = "No QSMs."
    MESSAGE_ADDED_FILES: str = "Added {} QSMs."
    MESSAGE_SELECTED_FILES: str = "Selected {} QSMs."

    def __init__(self):
        self._root = tk.Tk()
        self._root.title("Parameter exporter")
        self._root.geometry("640x720")
        self._load_translations()
        self._tree_dataframe = pd.DataFrame()
        self._name_to_branch_dataframe = {}

        tk.Button(self._root, text="About", command=lambda: showinfo(title="About", message=TIP_ABOUT)).pack(fill="x")
        
        ttk.Separator(self._root, orient='horizontal').pack(fill="x", pady=5)
        
        tk.Label(self._root, text="Import QSMs:").pack(fill="x")
        tk.Button(self._root, text="Add QSMs", command=self._open_files).pack(fill="x")

        file_list_frame = tk.Frame(self._root)
        file_list_frame.pack(fill="both", expand=True)

        file_list_box_vertical_scrollbar = tk.Scrollbar(file_list_frame, orient="vertical")
        file_list_box_vertical_scrollbar.pack(side="right", fill="y")

        self._file_list_variable = tk.Variable(value=list())
        self._file_list_variable.trace("w", self._update_file_summary)
        self._file_list_box = tk.Listbox(file_list_frame, listvariable=self._file_list_variable, selectmode=tk.MULTIPLE,  yscrollcommand=file_list_box_vertical_scrollbar.set)
        self._file_list_box.pack(fill="both", expand=True)
        self._file_list_box.bind('<<ListboxSelect>>', self._update_file_summary)
        file_list_box_vertical_scrollbar.config(command=self._file_list_box.yview)

        self._file_summary_string_var = tk.StringVar(value=App.MESSAGE_NO_FILES)
        self._file_overview_entry = tk.Entry(self._root, textvariable=self._file_summary_string_var, state="readonly")
        self._file_overview_entry.pack(fill="x")

        tk.Button(self._root, text="Remove selected QSMs", command=self._remove_selected_files).pack(fill="x")
        tk.Button(self._root, text="Clear list", command=self._clear_file_list).pack(fill="x")

        ttk.Separator(self._root, orient='horizontal').pack(fill="x", pady=5)

        tk.Label(self._root, text="Select language (Optional):").pack(fill="x")

        language_frame = tk.Frame(self._root)
        language_frame.pack(fill="x")

        language_list_box_vertical_scrollbar = tk.Scrollbar(language_frame, orient="vertical")
        language_list_box_vertical_scrollbar.pack(side="right", fill="y")

        self._language_list_variable = tk.Variable(value=list(self._language_to_translations.keys())+[""])
        self._language_list_box = tk.Listbox(language_frame, listvariable=self._language_list_variable, height=4, selectmode=tk.SINGLE,  yscrollcommand=language_list_box_vertical_scrollbar.set)
        self._language_list_box.pack(fill="x")
        language_list_box_vertical_scrollbar.config(command=self._language_list_box.yview)

        ttk.Separator(self._root, orient='horizontal').pack(fill="x", pady=5)


        self._int_var_only_output_tree_parameters = tk.IntVar(value=0)
        tk.Checkbutton(self._root, text="Only output tree parameters", variable=self._int_var_only_output_tree_parameters).pack(anchor="w")

        
        self._int_var_auto_open_result = tk.IntVar(value=1)
        tk.Checkbutton(self._root, text="Automatically open after export completion", variable=self._int_var_auto_open_result).pack(anchor="w")
        
        tk.Button(self._root, text="Save to", command=self._export_parameters).pack(fill="x")
        
        self._root.mainloop()
        return
    
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
        return

    def _load_translations(self):
        try:
            with open(os.path.join(APP_DIR, "qsm_viewer_translations.yaml"), "r", encoding="utf-8") as f:
                self._language_to_translations = yaml.safe_load(f)
        except Exception:
            self._language_to_translations = {}
        return

    def _open_files(self):
        paths = askopenfilenames(
            title="Open",
            filetypes=[
                ('QSM', '*.mat')
            ]
        )
        if not paths:
            pass
        opened_paths = list(self._file_list_variable.get())
        for path in paths:
            try:
                if path in opened_paths:
                    showwarning(title="Warning", message="File {} is already opened.".format(path))
                    continue

                qsm = loadmat(path)
                
                treedata = qsm["QSM"][0, 0]["treedata"][0, 0]
                branch = qsm["QSM"][0, 0]["branch"][0, 0]
                
            except Exception as e:
                showerror(title="Error", message="Failed to load file {}: {}".format(path, e))
                continue
            tree_parameter_to_value = {}
            for field in treedata.dtype.names:
                tree_parameter_to_value[field] = [treedata[field][0, 0]]

            name = Path(path).stem
            if name.lower().endswith("_qsm"):
                name = name[:-4]
            
            if "name" in self._tree_dataframe.columns:
                if name in self._tree_dataframe["name"].values:
                    i = 2
                    while True:
                        if name + f" ({i})" not in self._tree_dataframe["name"].values:
                            name = name + f" ({i})"
                            break
                        i += 1
            tree_parameter_to_value = {
                "__name__": name,
                "__path__": path, # Make these two at the beginning
                **tree_parameter_to_value
            } 
            self._tree_dataframe = pd.concat([self._tree_dataframe, pd.DataFrame(tree_parameter_to_value)], ignore_index=True, sort=False)
            
            
            branch_parameter_to_values = {}
            length = None
            for field in branch.dtype.names:
                values = branch[field]
                if isinstance(values, np.ndarray) and values.dtype == np.uint8:
                    if np.all(np.isin(values, [0, 1])):
                        values = values.astype(np.bool_)
                values = values.ravel()
                branch_parameter_to_values[field] = np.squeeze(values).tolist() 
                if length is None:
                    length = len(values)
                elif len(values) != length:
                    raise ValueError(f"Length of field {field} ({len(values)}) is different from length of other fields ({length})")
                branch_parameter_to_values[field] = values
            self._name_to_branch_dataframe[name] = pd.DataFrame(branch_parameter_to_values)
            
            opened_paths.append(path)

        self._file_list_variable.set(opened_paths)
    def _remove_selected_files(self):
        selected_indices = self._file_list_box.curselection()
        if selected_indices:
            file_list = list(self._file_list_variable.get())
            for idx in sorted(selected_indices, reverse=True):
                if  0 <= idx < len(file_list):
                    path = file_list[idx]
                    name = self._tree_dataframe[self._tree_dataframe["__path__"] == path]["__name__"].values[0]
                    self._tree_dataframe = self._tree_dataframe[self._tree_dataframe["__path__"] != path]
                    del self._name_to_branch_dataframe[name]
                    del file_list[idx]
            self._file_list_variable.set(file_list)
            self._file_list_box.selection_clear(0, tk.END)
    
    def _clear_file_list(self):
        self._file_list_variable.set(())
        self._tree_dataframe = pd.DataFrame()
        self._name_to_branch_dataframe.clear()

    def _export_parameters(self):
        path = asksaveasfilename(
            title="Save to",
            defaultextension=".xlsx",
            filetypes=[
                ('Excel workbook', '*.xlsx')
            ]
        )
        if not path:
            return

        sheet_name_to_dataframe = {"__index__": self._tree_dataframe}
        if not bool(self._int_var_only_output_tree_parameters.get()):
            sheet_name_to_dataframe.update(self._name_to_branch_dataframe)
        
        language_list_idx = self._language_list_box.curselection()
        language = ""
        if language_list_idx:
            language = self._language_list_box.get(language_list_idx[0])
        try:
            with xlsxwriter.Workbook(path) as workbook:
                for i, (sheet_name, dataframe) in enumerate(sheet_name_to_dataframe.items()):
                    worksheet = workbook.add_worksheet(sheet_name)

                    original_headers = list(dataframe.columns)
                    headers = []
                    for header in original_headers:
                        try:
                            headers.append(
                                self._language_to_translations[language][header]
                            )
                        except Exception:
                            headers.append(header)

                    data = dataframe.values.tolist()
                    data.insert(0, headers)

                    for row_num, row_data in enumerate(data):
                        worksheet.write_row(row_num, 0, row_data)
        except xlsxwriter.exceptions.FileCreateError:
            showerror(title="Error", message="Failed to create file: File is in use.")
            return
        
        if self._int_var_auto_open_result.get():
            os.startfile(path)
        
        showinfo(title="Info", message="Parameters successfully exported to {}".format(path))

def main():
    App()

if __name__ == "__main__":
    check_update()
    main()
