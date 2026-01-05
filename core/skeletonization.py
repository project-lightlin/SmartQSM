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

from .pipeline import Pipeline
from typing import Any, Generator
from .segmentation_based_skeletonization import SegmentationBasedSkeletonization
from .thinning_based_skeletonization import ThinningBasedSkeletonization
from .skeletonization_base import SkeletonizationBase

class Skeletonization:
    _skeletonizer: SkeletonizationBase

    def __init__(self, verbose: bool = False) -> None:
        self._verbose = verbose
        return
    
    def set_params(self, category: str = "thinning", **kwargs) -> None:
        if category == "segmentation":
            self._skeletonizer = SegmentationBasedSkeletonization(verbose=self._verbose)
        elif category == "thinning":
            self._skeletonizer = ThinningBasedSkeletonization(verbose=self._verbose)
        else:
            raise ValueError(f"Unknown category: {category}")
        self._skeletonizer.set_params(**kwargs)
        return
    
    def __len__(self) -> int:
        return len(self._skeletonizer)
    
    def run(self, **kwargs) -> Generator[Any, None, Any]:
        return self._skeletonizer.run(**kwargs)