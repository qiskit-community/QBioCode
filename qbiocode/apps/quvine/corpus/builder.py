# Copyright 2026, IBM Corporation.
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

from qbiocode.apps.quvine.corpus.operations import compile_corpus 
from typing import List
class CorpusBuilder: 
    def __init__(self):
        self._node_corpus = {} 
    
    # def add(self, root, view_walks): 
    #     self._node_corpus[root] = define_corpus(view_walks)
    def add(self, root, view_walks):
        # view_walks: List[List[str]]
        assert isinstance(view_walks, list)
        assert all(isinstance(w, list) for w in view_walks)
        
        self._node_corpus[root] = view_walks
        
    def build(self) -> List[List[str]]:
        corpus = compile_corpus(self._node_corpus)

        # Hard safety check
        for i, w in enumerate(corpus):
            if not isinstance(w, list):
                raise RuntimeError(f"Bad corpus entry {i}: {type(w)}")
            for t in w:
                if not isinstance(t, str):
                    raise RuntimeError(f"Bad token {t}: {type(t)}")

        return corpus
