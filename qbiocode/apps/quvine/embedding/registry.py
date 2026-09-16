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

class EmbeddingStore: 
    def __init__(self): 
        self._Z = {} 
        
    def add(self, name, Z): 
        self._Z[name] = Z 
    
    def names(self): 
        return list(self._Z.keys())
    
    def items(self): 
        return self._Z.items() 
    
    def get(self, name): 
        return self._Z[name]
    
    def as_list(self): 
        return list(self._Z.values())
    
    
