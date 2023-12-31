# Copyright 2021 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch


def gdt(p1, p2, mask, cutoffs):
    """
        Compute GDT between two structures.
        (Global Distance Test under specified distance cutoff)
        Args:
            p1:
                [*, N, 3] superimposed predicted (ca) coordinate tensor
            p2:
                [*, N, 3] ground-truth (ca) coordinate tensor
            mask:
                [*, N] residue masks
            cutoffs:
                A tuple of size 4, which contains distance cutoffs.
        Returns:
            A [*] tensor contains the final GDT score.
    """
    n = torch.sum(mask, dim=-1) # [*]
    
    p1 = p1.float()
    p2 = p2.float()
    distances = torch.sqrt(torch.sum((p1 - p2)**2, dim=-1)) # [*, N]
    
    scores = []
    for c in cutoffs:
        score = torch.sum((distances <= c) * mask, dim=-1) / n # [*]
        scores.append(score)

    return sum(scores) / len(scores)


def gdt_ts(p1, p2, mask):
    return gdt(p1, p2, mask, [1., 2., 4., 8.])


def gdt_ha(p1, p2, mask):
    return gdt(p1, p2, mask, [0.5, 1., 2., 4.])

