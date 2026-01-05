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

import sys
import os
import platform
import locale
import open3d.visualization.gui as gui
import tkinter
import numpy as np
from scipy.spatial.transform import Rotation
from _updater import check_update

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
sys.path.append(ROOT_DIR)
IS_MAC_OS = platform.system() == "Darwin"

def set_default_font_description():
    font_description = gui.FontDescription()
    locales = locale.getdefaultlocale()
    default_locale = locales[0].lower() if locales and locales[0] else "en_us"  
    if platform.system() == "Windows":
        if default_locale.startswith("ja"):
            font_description.add_typeface_for_language("YuGothM", "ja")
        font_description.add_typeface_for_language("malgun", "ko")
        font_description.add_typeface_for_language("LeelawUI", "th")
        font_description.add_typeface_for_language("segoeui", "vi")
        font_description.add_typeface_for_language("msyh", "zh_all")
    elif platform.system() == "Darwin":
        if default_locale.startswith("ja"):
            font_description.add_typeface_for_language("ヒラギノ角ゴシック W3", "ja")
        font_description.add_typeface_for_language("AppleSDGothicNeo", "ko")
        font_description.add_typeface_for_language("Thonburi", "th")
        font_description.add_typeface_for_language("Arial", "vi")
        font_description.add_typeface_for_language("PingFang", "zh_all")
    else:  # Linux
        if default_locale.startswith("ja"):
            font_description.add_typeface_for_language("NotoSansCJK-Regular", "ja")
        font_description.add_typeface_for_language("NotoSansCJK-Regular", "ko")
        font_description.add_typeface_for_language("NotoSansThai-Regular", "th")
        font_description.add_typeface_for_language("NotoSans-Regular", "vi")
        font_description.add_typeface_for_language("NotoSansCJK-Regular", "zh_all")
    gui.Application.instance.set_font(0, font_description)
    return font_description

def convert_ndc_to_world_ray(camera, view_width, view_height, mouse_x, mouse_y):
    world_near = camera.unproject(mouse_x, mouse_y, -0.999, view_width, view_height) #-1 or 1 could cause the result to be inf 
    world_far  = camera.unproject(mouse_x, mouse_y, 0.999, view_width, view_height)
    origin = np.asarray(world_near).reshape(3)
    direction = np.asarray(world_far).reshape(3) - origin

    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return origin, np.array([0, 0, 0], dtype=np.float64)
    return origin, direction / norm

def get_screen_resolution():
    root = tkinter.Tk()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.destroy()
    return width, height

def get_orthogonal_up(look_dir, world_up=np.array([0.,0.,1.])):
    look_dir = look_dir / np.linalg.norm(look_dir)
    if abs(np.dot(look_dir, world_up)) > 0.999999:
        world_up = np.array([0.,1.,0.])
    right = np.cross(world_up, look_dir)
    right /= np.linalg.norm(right)
    up = np.cross(look_dir, right)
    return up / np.linalg.norm(up)

def compute_new_eye(eye, dir, off, dir_new):
    u = dir / np.linalg.norm(dir)
    v = dir_new / np.linalg.norm(dir_new)
    
    C = eye + off
    d = np.linalg.norm(off)
    
    up_old = get_orthogonal_up(u)
    right_old = np.cross(up_old, u)  
    right_old /= np.linalg.norm(right_old)
    
    x0 = np.dot(off, right_old)
    y0 = np.dot(off, up_old)
    
    
    up_new = get_orthogonal_up(v)
    right_new = np.cross(up_new, v)  
    right_new /= np.linalg.norm(right_new)
    
    t_sq = d**2 - (x0**2 + y0**2)
    if t_sq < 0:
        raise ValueError("The original screen coordinate distance exceeds the distance from the camera to the object.")
    t = np.sqrt(t_sq) 
    
   
    off_new = t * v + x0 * right_new + y0 * up_new
    
    eye_new = C - off_new
    return eye_new  

def show_multibutton_messagebox(window, title, message, button_text_to_callback, arrangement="horizontal"):
    dialog = gui.Dialog(title)
    dialog_layout = gui.Vert(5, gui.Margins(5, 5, 5, 5))
    dialog_layout.add_child(gui.Label(message))

    if arrangement == "vertical":
        button_layout = gui.Vert(5)
    elif arrangement == "horizontal":
        button_layout = gui.Horiz(5)
    else:
        raise NotImplementedError(f"Unknown arrangement: {arrangement}")
        
    button_layout.add_stretch()
    for button_text, callback in button_text_to_callback.items():
        button = gui.Button(button_text)
        button.set_on_clicked(callback)
        button_layout.add_child(button)
    button_layout.add_stretch()
    dialog_layout.add_child(button_layout)
    dialog.add_child(dialog_layout)
    window.show_dialog(dialog)
    return

def show_snackbar(window, message):
    em = window.theme.font_size
    dialog = gui.Dialog("") # Title does not display
    dialog_layout = gui.Vert(em, gui.Margins(em, em, em, em))
    dialog_layout.add_child(gui.Label(message))
    dialog.add_child(dialog_layout)
    window.show_dialog(dialog)
