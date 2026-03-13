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

from typing import Dict
from .core_algorithm_base import CoreAlgorithmBase
from .spconv_based_contraction import SpconvBasedContraction
from .layerwise_clustering import LayerwiseClustering
from .space_colonization import SpaceColonization

registry: Dict[str, Dict[str, CoreAlgorithmBase]] = {
    "thinning": {
        "spconv_based_contraction": SpconvBasedContraction
    },
    "segmentation": {
        "layerwise_clustering": LayerwiseClustering,
        "space_colonization": SpaceColonization
    }
}