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

from typing import Dict, Callable, Any, List
import numpy as np

class CoreAlgorithmBase:
    _points: np.ndarray
    _verbose: bool
    
    def __init__(self, *, verbose: bool = False, **kwargs) -> None:
        self._verbose = verbose
        return
    
    def get_pipeline(self) -> List[Callable[[], Any]]:
        pass

    def set_points(self, points: np.ndarray) -> None:
        self._points = points
        return

    def output(self) -> Any:
        pass