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

__all__ = ("logger", "init_logger")

import logging

logging_levels = {
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL,
    'debug': logging.DEBUG
}


class LoggerError(Exception):
    def __init__(self, arg):
        super().__init__(f"unknown log level '{arg}'")


handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(fmt="%(levelname)s: %(message)s"))

logger = logging.getLogger("operator")
logger.propagate = False
logger.addHandler(handler)


def init_logger(level):
    if level not in logging_levels.keys():
        raise LoggerError(level)
    logger.setLevel(logging_levels[level])
