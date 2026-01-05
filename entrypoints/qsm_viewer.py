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

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import sys
import os
from scipy.io import loadmat
from scipy.io.matlab._miobase import MatReadError
import yaml
from _common_import import *
from utils.hash import get_md5_hash
import pandas as pd
import pyperclip
import traceback
from tkinter.filedialog import askopenfilename
import threading

APP_DIR = os.path.dirname(os.path.abspath(__file__))

class App:
    MENU_QUIT = 1
    MENU_CHECK_UPDATE = 2
    MENU_CHANGE_LANGUAGE = 3
    MENU_ABOUT = 4
    MENU_OPEN_FILE = 11
    MENU_TOP_VIEW = 211
    MENU_FRONT_VIEW = 212
    MENU_LEFT_VIEW = 213
    MENU_BACK_VIEW = 214
    MENU_RIGHT_VIEW = 215
    MENU_BOTTOM_VIEW = 216
    MENU_FRONT_ISOMETRIC_VIEW = 217
    MENU_BACK_ISOMETRIC_VIEW = 218
    MENU_ZOOM_TO_FIT = 221


    def __init__(self):
        max_width, max_height = get_screen_resolution()
        min_width, min_height = 1280, 720

        config = {
            "width": min_width,
            "height": min_height,
            "language": "English"
        }
        try:
            with open(os.path.join(APP_DIR, "qsm_viewer_config.yaml"), "r", encoding="utf-8") as f:
                config.update(yaml.safe_load(f))
        except Exception:
            pass
        
        # Check validity -- Start
        config["language"] = str(config["language"])
        try:
            config["width"] = int(config["width"])
            config["height"] = int(config["height"])
        except Exception:
            config["width"] = min_width
            config["height"] = min_height
        if not (
            min(min_width, max_width) <= config["width"] <= max(min_width, max_width) \
            and \
            min(min_height, max_height) <= config["height"] <= max(min_height, max_height)
        ):
            config["width"] = min_width
            config["height"] = min_height
      
        # Check validity -- End

        self.window = gui.Application.instance.create_window("QSM Viewer", config["width"], config["height"])
        
        self._translation = {}
        self._language = config.get("language", "English")
        self._supported_languages = []
        self._load_translation()
        self._tree_dict = {}
        self._branch_dataframe = pd.DataFrame()

        # menubar
        if gui.Application.instance.menubar is None:
            self._menu = gui.Menu()

            if IS_MAC_OS:
                app_menu = gui.Menu()
                self._menu.add_menu("QSM Viewer", app_menu)

                app_menu.add_item(
                    self._translation.get("change_language", "Change language [Current English]"), 
                    App.MENU_CHANGE_LANGUAGE
                )
                app_menu.add_item(
                    self._translation.get("check_update", "Check update"), 
                    App.MENU_CHECK_UPDATE
                )
                app_menu.add_item(
                    self._translation.get("about", "About"), 
                    App.MENU_ABOUT
                )
                app_menu.add_separator()
                app_menu.add_item(
                    self._translation.get("quit", "Quit"), 
                    App.MENU_QUIT
                )
            
            file_menu = gui.Menu()
            self._menu.add_menu(
                self._translation.get("file", "File"), 
                file_menu
            )

            file_menu.add_item(
                self._translation.get("open_file", "Open file") + "(Ctrl+O)", 
                App.MENU_OPEN_FILE
            )
            if not IS_MAC_OS:
                file_menu.add_separator()
                file_menu.add_item(
                    self._translation.get("quit", "Quit"), 
                    App.MENU_QUIT
                )
            
            display_menu = gui.Menu()
            self._menu.add_menu(
                self._translation.get("display", "Display"), 
                display_menu
            )

            display_menu.add_item(
                self._translation.get("front_view", "Front view") + "(Enter | Ctrl+1)", 
                App.MENU_FRONT_VIEW
            )
            display_menu.add_item(
                self._translation.get("back_view", "Back view") + "(Backspace | Ctrl+2)", 
                App.MENU_BACK_VIEW
            )
            display_menu.add_item(
                self._translation.get("left_view", "Left view") + "(Left | Ctrl+3)", 
                App.MENU_LEFT_VIEW
            )
            display_menu.add_item(
                self._translation.get("right_view", "Right view") + "(Right | Ctrl+4)", 
                App.MENU_RIGHT_VIEW
            )
            display_menu.add_item(
                self._translation.get("top_view", "Top view") + "(Up | Ctrl+5)", 
                App.MENU_TOP_VIEW
            )
            display_menu.add_item(
                self._translation.get("bottom_view", "Bottom view") + "(Down | Ctrl+6)", 
                App.MENU_BOTTOM_VIEW
            )
            display_menu.add_item(
                self._translation.get("front_isometric_view", "Front isometric view") + "(Pageup | 7)", 
                App.MENU_FRONT_ISOMETRIC_VIEW
            )
            display_menu.add_item(
                self._translation.get("back_isometric_view", "Back (Standard) isometric view") + "(Pagedown | Ctrl+7)", 
                App.MENU_BACK_ISOMETRIC_VIEW
            )
            display_menu.add_separator()
            display_menu.add_item(
                self._translation.get("zoom_to_fit", "Zoom to fit") + "(Space)", 
                App.MENU_ZOOM_TO_FIT
            )
            

            if not IS_MAC_OS:
                display_menu.add_separator()
                display_menu.add_item(
                    self._translation.get("change_language", "Change language (Current English)"), 
                    App.MENU_CHANGE_LANGUAGE
                )

                help_menu = gui.Menu()
                self._menu.add_menu(
                    self._translation.get("help", "Help"), 
                    help_menu
                )

                help_menu.add_item(
                    self._translation.get("check_update", "Check update"), 
                    App.MENU_CHECK_UPDATE
                )
                help_menu.add_item(
                    self._translation.get("about", "About"), 
                    App.MENU_ABOUT
                )

            gui.Application.instance.menubar = self._menu
        self.window.set_on_menu_item_activated(App.MENU_OPEN_FILE, self._on_menu_open_file_activated)
        self.window.set_on_menu_item_activated(App.MENU_QUIT, self._on_close)
        self.window.set_on_menu_item_activated(App.MENU_TOP_VIEW, self._on_menu_top_view_activated)
        self.window.set_on_menu_item_activated(App.MENU_FRONT_VIEW, self._on_menu_front_view_activated)
        self.window.set_on_menu_item_activated(App.MENU_LEFT_VIEW, self._on_menu_left_view_activated)
        self.window.set_on_menu_item_activated(App.MENU_BACK_VIEW, self._on_menu_back_view_activated)
        self.window.set_on_menu_item_activated(App.MENU_RIGHT_VIEW, self._on_menu_right_view_activated)
        self.window.set_on_menu_item_activated(App.MENU_BOTTOM_VIEW, self._on_menu_bottom_view_activated)
        self.window.set_on_menu_item_activated(App.MENU_FRONT_ISOMETRIC_VIEW, self._on_menu_front_isometric_view_activated)
        self.window.set_on_menu_item_activated(App.MENU_BACK_ISOMETRIC_VIEW, self._on_menu_back_isometric_view_activated)
        self.window.set_on_menu_item_activated(App.MENU_ZOOM_TO_FIT, self._on_menu_zoom_to_fit_activated)
        self.window.set_on_menu_item_activated(App.MENU_CHANGE_LANGUAGE, self._on_menu_change_language_activated)
        
        self.window.set_on_menu_item_activated(App.MENU_CHECK_UPDATE, self._on_menu_check_update_activated)
        self.window.set_on_menu_item_activated(App.MENU_ABOUT, self._on_menu_about_activated)


        # scene
        self._scene_widget = gui.SceneWidget()
        self._scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self._scene_widget.set_on_key(self._on_key_event)
        self._scene_widget.set_on_mouse(self._on_mouse_event)
        self._scene_widget.scene.set_lighting(self._scene_widget.scene.LightingProfile.MED_SHADOWS, np.array([0., 0., -1.]))
        self.window.add_child(self._scene_widget)

        # panel
        self._vert = gui.Vert(
            5, 
            gui.Margins(5, 5, 5, 5)
        )
        self._label = gui.Label(
            self._translation.get("parameters", "Parameters (Click to copy to clipboard)")
        )
        self._tree_view = gui.TreeView()
        self._tree_view.can_select_items_with_children = True
        self._tree_view.set_on_selection_changed(self._on_tree_view_selection_changed)
        self._vert.add_child(self._label)
        self._vert.add_child(self._tree_view)
        self.window.add_child(self._vert)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)

        # Initialize
        self._raycasting_scene = o3d.t.geometry.RaycastingScene()
        self._pressed_keys = set()
        self._pressed_mouse_button_to_repetitive = {}
        self._item_id_to_value = {}
        self._thread = None
        self._is_first_load = True

    def _on_layout(self, layout_context):
        content_rect = self.window.content_rect
        width = 17 * layout_context.theme.font_size
        height = content_rect.height
        self._scene_widget.frame = gui.Rect(
            content_rect.x,
            content_rect.y,
            content_rect.get_right() - width,
            height
        )
        self._vert.frame = gui.Rect(
            content_rect.get_right() - width, 
            content_rect.y, 
            width,
            height
        )
        
    def _on_menu_open_file_activated(self):
        if self._thread is not None and self._thread.is_alive():
            return
        # gui.FileDialog in Open3D has significant flaws
        
        def select_and_load_file():
            path = askopenfilename(
                title=self._translation.get("open", "Open"),
                filetypes=[("QSM", "*.mat")]
            )
            gui.Application.instance.post_to_main_thread(self.window, self.window.close_dialog)
            if path:
                gui.Application.instance.post_to_main_thread(self.window, lambda: self.load_qsm(path))
        #Disabling the control doesn't seem to work, interrupting any response to the window using a dialog without buttons.
        show_snackbar(self.window, self._translation.get("waiting", "Waiting...")) 
        self._thread = threading.Thread(target=select_and_load_file)
        self._thread.start()


    def load_qsm(self, path):
        self._dehighlight_and_hide_parameters()
        self._scene_widget.scene.clear_geometry()
        self._raycasting_scene = None
        self._tree_dict = {}
        self._branch_dataframe = None
        self._mesh = None

        try:
            qsm = loadmat(path)

            mesh_basename = qsm["QSM"][0, 0]["rundata"][0, 0]["target_ply"][0]
            mesh_path = os.path.join(os.path.dirname(path), mesh_basename)
            if not os.path.exists(mesh_path):
                raise FileNotFoundError
            real_mesh_hash = qsm["QSM"][0, 0]["rundata"][0, 0]["hash"][0]
            mesh_hash = get_md5_hash(mesh_path)
            if real_mesh_hash != mesh_hash:
                raise AssertionError

            self._mesh = o3d.io.read_triangle_mesh(mesh_path) # Cannot read vertex and face normal vectors
            aabb = self._mesh.get_axis_aligned_bounding_box()
            global_shift = np.array([
                -(aabb.min_bound[0] + aabb.max_bound[0]) / 2.0,
                -(aabb.min_bound[1] + aabb.max_bound[1]) / 2.0,
                -aabb.min_bound[2]
            ], dtype = np.float64)
            self._mesh.translate(global_shift)
            self._mesh.compute_vertex_normals()
            material_record = rendering.MaterialRecord()
            material_record.shader = "defaultLit"

            treedata = qsm["QSM"][0, 0]["treedata"][0, 0]
            for field in treedata.dtype.names:
                self._tree_dict[field] = treedata[field][0, 0]

            branch = qsm["QSM"][0, 0]["branch"][0, 0]
            branch_parameter_to_values = {}
            branch_parameter_to_dtype = {}
            for field in branch.dtype.names:
                values = branch[field]
                if isinstance(values, np.ndarray) and values.dtype == np.uint8:
                    if np.all(np.isin(values, [0, 1])):
                        values = values.astype(np.bool_)
                branch_parameter_to_values[field] = np.squeeze(values).tolist()
                branch_parameter_to_dtype[field] = values.dtype
            self._branch_dataframe = pd.DataFrame(branch_parameter_to_values).astype(branch_parameter_to_dtype)

            self._scene_widget.scene.add_geometry("__model__", self._mesh, material_record)
            self._raycasting_scene = o3d.t.geometry.RaycastingScene()
            self._raycasting_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(self._mesh))

            self._on_menu_zoom_to_fit_activated()
            if self._is_first_load:
                self._on_menu_front_view_activated()
                self._is_first_load = False
        except MatReadError:
            self.window.show_message_box(
                "", # Title does not display
                self._translation.get("qsm_file_load_failed", "Failed to load QSM file {}.").format(path)
            )
        except FileNotFoundError:
            self.window.show_message_box(
                "", # Title does not display
                self._translation.get("ply_file_not_found", "Unable to find associated PLY file {}. Please ensure that this file is in the same directory as the QSM file.").format(mesh_path)
            )
        except AssertionError:
            self.window.show_message_box(
                "", # Title does not display
                self._translation.get("ply_file_hash_mismatch", "Mismatched file hash value. PLY file {} has been illegally modified or damaged.").format(mesh_path)
            )
        except Exception:
            traceback.print_exc()
            self.window.show_message_box(
                "", # Title does not display
                self._translation.get("unsupported_qsm_file_format", "Unsupported QSM file format in file {}.").format(path)
            )
        return
    
    def _load_translation(self):
        try:
            with open(os.path.join(APP_DIR, "qsm_viewer_translations.yaml"), "r", encoding="utf-8") as f:
                language_to_translations = yaml.safe_load(f)
                self._translation = language_to_translations["English"] if "English" in language_to_translations else {} # Dynamically read external translations, otherwise a lot of repetitive code needs to be added.
                self._translation.update(language_to_translations[self._language])
                self._supported_languages = list(set(list(language_to_translations.keys())) | set(["English"]))
        except Exception:
            self._language = "English"
            self._supported_languages = ["English"]
        return
    
    def _on_close(self):
        try:
            size = self.window.size
            config = {
                "width": size.width,
                "height": size.height,
                "language": self._language
            }
            with open(os.path.join(APP_DIR, "qsm_viewer_config.yaml"), "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, allow_unicode=True)
        except Exception:
            pass
        if self._thread is not None and self._thread.is_alive():
            self._thread._stop()
        gui.Application.instance.quit()

    def _on_key_event(self, event):
        if event.type == gui.KeyEvent.DOWN:
            self._pressed_keys.add(event.key)
            return gui.SceneWidget.EventCallbackResult.HANDLED
        elif event.type == gui.KeyEvent.UP:
            if (set([gui.KeyName.LEFT_CONTROL, gui.KeyName.O]) == self._pressed_keys or set([gui.KeyName.RIGHT_CONTROL, gui.KeyName.O]) == self._pressed_keys) and event.key == gui.KeyName.O: # Ctrl + O
                self._on_menu_open_file_activated()
            elif set([gui.KeyName.UP]) == self._pressed_keys or (
                (set([gui.KeyName.LEFT_CONTROL, gui.KeyName.FIVE]) == self._pressed_keys or set([gui.KeyName.RIGHT_CONTROL, gui.KeyName.FIVE]) == self._pressed_keys) and event.key == gui.KeyName.FIVE
            ):
                self._on_menu_top_view_activated()
            elif set([gui.KeyName.DOWN]) == self._pressed_keys or (
                (set([gui.KeyName.LEFT_CONTROL, gui.KeyName.SIX]) == self._pressed_keys or set([gui.KeyName.RIGHT_CONTROL, gui.KeyName.SIX]) == self._pressed_keys) and event.key == gui.KeyName.SIX
            ):
                self._on_menu_bottom_view_activated()
            elif set([gui.KeyName.LEFT]) == self._pressed_keys or (
                (set([gui.KeyName.LEFT_CONTROL, gui.KeyName.THREE]) == self._pressed_keys or set([gui.KeyName.RIGHT_CONTROL, gui.KeyName.THREE]) == self._pressed_keys) and event.key == gui.KeyName.THREE
            ):
                self._on_menu_left_view_activated()
            elif set([gui.KeyName.RIGHT]) == self._pressed_keys  or (
                (set([gui.KeyName.LEFT_CONTROL, gui.KeyName.FOUR]) == self._pressed_keys or set([gui.KeyName.RIGHT_CONTROL, gui.KeyName.FOUR]) == self._pressed_keys) and event.key == gui.KeyName.FOUR
            ):
                self._on_menu_right_view_activated()
            elif set([gui.KeyName.ENTER]) == self._pressed_keys  or (
                (set([gui.KeyName.LEFT_CONTROL, gui.KeyName.ONE]) == self._pressed_keys or set([gui.KeyName.RIGHT_CONTROL, gui.KeyName.ONE]) == self._pressed_keys) and event.key == gui.KeyName.ONE
            ):
                self._on_menu_front_view_activated()
            elif set([gui.KeyName.BACKSPACE]) == self._pressed_keys or (
                (set([gui.KeyName.LEFT_CONTROL, gui.KeyName.TWO]) == self._pressed_keys or set([gui.KeyName.RIGHT_CONTROL, gui.KeyName.TWO]) == self._pressed_keys) and event.key == gui.KeyName.TWO
            ):
                self._on_menu_back_view_activated()
            elif set([gui.KeyName.PAGE_UP]) == self._pressed_keys or set([gui.KeyName.SEVEN]) == self._pressed_keys:
                self._on_menu_front_isometric_view_activated()
            elif set([gui.KeyName.PAGE_DOWN]) == self._pressed_keys or (
                (set([gui.KeyName.LEFT_CONTROL, gui.KeyName.SEVEN]) == self._pressed_keys or set([gui.KeyName.RIGHT_CONTROL, gui.KeyName.SEVEN]) == self._pressed_keys) and event.key == gui.KeyName.SEVEN
            ):
                self._on_menu_back_isometric_view_activated()
            elif set([gui.KeyName.SPACE]) == self._pressed_keys:
                self._on_menu_zoom_to_fit_activated()
            
            self._pressed_keys.discard(event.key)
            return gui.SceneWidget.EventCallbackResult.HANDLED
        return gui.SceneWidget.EventCallbackResult.IGNORED


    def _set_view(self, forward):
        # Do not change view when there is nothing in the scene
        if self._scene_widget.scene.bounding_box.is_empty():
            return

        # Similar to CloudCompare, but not based on angle
        cam = self._scene_widget.scene.camera
        view_mat = cam.get_view_matrix()
        inv_view_mat = np.linalg.inv(view_mat)
        old_eye = inv_view_mat[:3, 3]
        old_forward = -inv_view_mat[:3, 2]
        model_center = self._scene_widget.scene.bounding_box.get_center()
        forward = forward / np.linalg.norm(forward)
        new_eye = compute_new_eye(
            old_eye,
            old_forward,
            model_center - old_eye,
            forward
        )
        up = get_orthogonal_up(forward)
        self._scene_widget.look_at(new_eye + forward, new_eye, up)

    def _on_menu_top_view_activated(self):
        self._set_view(np.array([0, 0, -1]))

    def _on_menu_bottom_view_activated(self):
        self._set_view(np.array([0, 0, 1]))

    def _on_menu_front_view_activated(self):
        self._set_view(np.array([0, 1, 0]))

    def _on_menu_back_view_activated(self):
        self._set_view(np.array([0, -1, 0]))

    def _on_menu_left_view_activated(self):
        self._set_view(np.array([1, 0, 0]))

    def _on_menu_right_view_activated(self):
        self._set_view(np.array([-1, 0, 0]))

    def _on_menu_front_isometric_view_activated(self):
        self._set_view(np.array([np.sqrt(3)/2, np.sqrt(3)/2, -1.]))

    def _on_menu_back_isometric_view_activated(self):
        self._set_view(np.array([-np.sqrt(3)/2, -np.sqrt(3)/2, -1.]))

    def _on_menu_zoom_to_fit_activated(self):
        cam = self._scene_widget.scene.camera
        view_mat = cam.get_view_matrix()
        inv_view_mat = np.linalg.inv(view_mat)
        old_forward = -inv_view_mat[:3, 2]

        bounds = self._scene_widget.scene.bounding_box
        self._scene_widget.setup_camera(60, bounds, bounds.get_center())

        self._set_view(old_forward)

    def _on_mouse_event(self, event):
        if event.x >= self._vert.frame.x and event.x <= self._vert.frame.x + self._vert.frame.width and event.y >= self._vert.frame.y and event.y <= self._vert.frame.y + self._vert.frame.height:
            return gui.Widget.EventCallbackResult.HANDLED
        if event.type == gui.MouseEvent.BUTTON_DOWN:
            self._pressed_mouse_button_to_repetitive[event.buttons] = False
            return gui.Widget.EventCallbackResult.HANDLED
        elif event.type == gui.MouseEvent.BUTTON_UP:
            if len(self._pressed_mouse_button_to_repetitive) == 1 and int(gui.MouseButton.LEFT) in self._pressed_mouse_button_to_repetitive and not self._pressed_mouse_button_to_repetitive[int(gui.MouseButton.LEFT)]:
                ray_origin, ray_direction = convert_ndc_to_world_ray(
                    self._scene_widget.scene.camera,
                    self._scene_widget.frame.width, 
                    self._scene_widget.frame.height,
                    event.x, 
                    event.y - self._scene_widget.frame.y # Remove the height of the menubar
                )
                rays = o3d.core.Tensor([[
                    ray_origin[0], ray_origin[1], ray_origin[2], ray_direction[0], ray_direction[1], ray_direction[2]
                ]], dtype=o3d.core.Dtype.Float32)
                ans = self._raycasting_scene.cast_rays(rays)
                t_hit = float(ans["t_hit"].numpy()[0])
                if not np.isinf(t_hit):
                    self._dehighlight_and_hide_parameters()
                    triangle_id = int(ans["primitive_ids"].numpy()[0])
                    self._highlight_and_display_parameters(triangle_id)
                
            self._pressed_mouse_button_to_repetitive.pop(event.buttons, None)
            return gui.Widget.EventCallbackResult.HANDLED
        elif event.type == gui.MouseEvent.DRAG:
            if event.buttons in self._pressed_mouse_button_to_repetitive:
                self._pressed_mouse_button_to_repetitive[event.buttons] = True
            return gui.Widget.EventCallbackResult.HANDLED

        return gui.Widget.EventCallbackResult.IGNORED
    
    def _highlight_and_display_parameters(self, triangle_id):
        vertices = np.asarray(self._mesh.vertices)
        triangles = np.asarray(self._mesh.triangles)
        vertex_colors = np.asarray(self._mesh.vertex_colors)

        record_idx = np.where((self._branch_dataframe["start"] <= triangle_id) & (self._branch_dataframe["end"] >= triangle_id))[0][0]
        triangle_ids = np.arange(
            self._branch_dataframe.iloc[record_idx]["start"], 
            self._branch_dataframe.iloc[record_idx]["end"] + 1, 
            dtype=type(triangle_id)
        )
        parent_branch_id = self._branch_dataframe.iloc[record_idx]["parent"]
        
        target_mesh = o3d.geometry.TriangleMesh()
        target_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        target_mesh.triangles = o3d.utility.Vector3iVector(triangles[triangle_ids])
        target_mesh.paint_uniform_color([0.25, 1., 1.])

        if record_idx == 0:
            for key, value in self._tree_dict.items():
                parameter_name = self._translation.get(key, key)
                item_id = self._tree_view.add_text_item(
                    self._tree_view.get_root_item(),
                    parameter_name
                )
                self._item_id_to_value[item_id] = parameter_name

                text_edit = gui.TextEdit()
                text_edit.text_value = str(value)
                text_edit.enabled = False
                item_id = self._tree_view.add_item(
                    item_id,
                    text_edit
                )
                self._item_id_to_value[item_id] = value
        else:
            for key, value in self._branch_dataframe.iloc[record_idx].items():
                if key in ["start", "end"]: 
                    continue
                parameter_name = self._translation.get(key, key)
                item_id = self._tree_view.add_text_item(
                    self._tree_view.get_root_item(),
                    parameter_name
                )
                self._item_id_to_value[item_id] = parameter_name
                
                text_edit = gui.TextEdit()
                text_edit.text_value = str(value)
                text_edit.enabled = False
                item_id = self._tree_view.add_item(
                    item_id,
                    text_edit
                )
                self._item_id_to_value[item_id] = value

        if parent_branch_id != 0:
            try:
                triangle_ids = np.arange(
                    self._branch_dataframe.loc[self._branch_dataframe["id"] == parent_branch_id, "start"].item(), 
                    self._branch_dataframe.loc[self._branch_dataframe["id"] == parent_branch_id, "end"].item() + 1,
                    dtype=type(triangle_id)
                )
                parent_mesh = o3d.geometry.TriangleMesh()
                parent_mesh.vertices = o3d.utility.Vector3dVector(vertices)
                parent_mesh.triangles = o3d.utility.Vector3iVector(triangles[triangle_ids])
                parent_mesh.paint_uniform_color(vertex_colors[triangles[triangle_ids[0]][0]])
                target_mesh += parent_mesh
            except Exception:
                # Future support for broken branches
                pass
        material_record = rendering.MaterialRecord()
        material_record.shader = "defaultUnlit"
        self._scene_widget.scene.add_geometry("highlight", target_mesh, material_record)

    def _dehighlight_and_hide_parameters(self):
        self._item_id_to_value.clear()
        self._tree_view.clear()
        self._scene_widget.scene.remove_geometry("highlight")
    
    def _on_tree_view_selection_changed(self, new_item_id):
        try:
            pyperclip.copy(f"{self._item_id_to_value[new_item_id]}")
        except Exception:
            pass
    

    def _on_menu_change_language_activated(self):
        em = self.window.theme.font_size
        dialog = gui.Dialog("")
        dialog_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        list_view = gui.ListView()
        list_view.set_items(self._supported_languages)
        list_view.set_max_visible_items(4)
        dialog_layout.add_child(list_view)

        horiz = gui.Horiz()
        horiz.add_stretch()
        button_ok = gui.Button("OK")

        def set_language():
            result = list_view.selected_value
            if result == "":
                self.window.close_dialog()
                self.window.show_message_box("", self._translation.get("no_language_selected", "No language selected."))
                return
            self._language = result
            self.window.close_dialog()
            self.window.show_message_box("", self._translation.get("setting_changed", "Please manually close the program for the changes to take effect."))
            self._menu.set_enabled(App.MENU_CHANGE_LANGUAGE, False)

        button_ok.set_on_clicked(
            set_language
        )
        horiz.add_child(button_ok)
        horiz.add_stretch()
        dialog_layout.add_child(horiz)

        dialog.add_child(dialog_layout)
        self.window.show_dialog(dialog)

    def _on_menu_check_update_activated(self):
        pass

    def _on_menu_about_activated(self):
        self.window.show_message_box("", self._translation.get("about_tree_viewer", "QSM Viewer is a part of the SmartQSM toolkit.\nFor details, please visit project-lightlin.github.io\nLicense: AGPL-3.0"))

def main():
    gui.Application.instance.initialize()
    set_default_font_description()
    app = App()
    if len(sys.argv) == 2:
        path = sys.argv[1]
        if os.path.exists(path):
            app.load_qsm(path)
        else:
            app.window.show_message_box(
                "", # Title does not display
                app._translation.get("file_not_exist", "File {} does not exist.").format(path)
            )
    elif len(sys.argv) > 2:
        app.window.show_message_box(
            "", # Title does not display
            app._translation.get("argument_error", "Invalid argument.\n\nusage: python qsm_viewer.py [QSM_FILE_PATH]\n\nStart GUI mode without any parameters.")
        )
    gui.Application.instance.run()

if __name__ == "__main__":
    check_update()
    main()