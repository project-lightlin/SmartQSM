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

from typing import Callable, List, Any, Generator

class Pipeline:
    _pipeline: List[Callable[[], Any]]
    _verbose: bool

    def __init__(self, verbose: bool = False) -> None:
        self._verbose = verbose
        self._pipeline = []
        return
    
    def _add_fns_to_pipeline(self, id: int, fns: List[Callable[[], Any]]) -> None:
        if id >= 0:
            id = min(id, len(self._pipeline)) #Eliminate potential order confusion
            for fn in reversed(fns):
                self._pipeline.insert(id, fn)
        else:
            raise IndexError(f"Negative index {id} is not supported to prevent ambiguity.")
        return

    def _clear_pipeline(self) -> None:
        self._pipeline.clear()
        return

    def __len__(self) -> int:
        return len(self._pipeline)
    
    def run(self, **kwargs) -> Generator[Any, None, Any]:
        pass
    
    def set_params(**kwargs) -> None:
        # Suggest controlling the pipeline within this function instead of the constructor to enhance flexibility
        pass