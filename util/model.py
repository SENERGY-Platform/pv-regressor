"""
   Copyright 2022 InfAI (CC SES)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

__all__ = ("OperatorConfig", )

import simple_struct
import json
import typing


class Selector(simple_struct.Structure):
    name: str = None
    args: typing.Set[str] = None

    def __init__(self, d, **kwargs):
        super().__init__(d, **kwargs)
        self.args = set(self.args)


class Config(simple_struct.Structure):
    logger_level = "warning"
    data_path = "/opt/data"
    selectors: typing.List[Selector] = None
    lat = 51.34
    long = 12.38

    def __init__(self, d, **kwargs):
        super().__init__(d, **kwargs)
        if self.selectors:
            self.selectors = [Selector(s) for s in json.loads(self.selectors)]


class Mapping(simple_struct.Structure):
    dest: str = None
    source: str = None


class InputTopic(simple_struct.Structure):
    name: str = None
    filterType: str = None
    filterValue: str = None
    mappings: typing.List[Mapping] = None

    def __init__(self, d, **kwargs):
        super().__init__(d, **kwargs)
        self.mappings = [Mapping(m) for m in self.mappings]


class OperatorConfig(simple_struct.Structure):
    config = Config
    inputTopics: typing.List[InputTopic] = None

    def __init__(self, d, **kwargs):
        super().__init__(d, **kwargs)
        self.inputTopics = [InputTopic(it) for it in self.inputTopics]
