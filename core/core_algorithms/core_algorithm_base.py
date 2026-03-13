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

from typing import Callable, Any, List, Dict

class CoreAlgorithmBase:
    _verbose: bool
    
    def __init__(self, *, verbose: bool = False, **kwargs) -> None:
        self._verbose = verbose
        return
    
    def get_pipeline(self) -> List[Callable[[], Any]]:
        pass

    def input(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, f"_{k}", v) # Initialize a new object attribute
        return

    def output(self) -> Dict[str, Any]:
        pass