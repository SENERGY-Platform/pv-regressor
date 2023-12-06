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

from ._util import *
import algo
import unittest


class TestOperator(unittest.TestCase):
    def test_route(self):
        mock_kafka_consumer = MockKafkaConsumer(mock_messages)
        mock_operator = MockOperator()
        mock_operator.init(
            kafka_consumer=mock_kafka_consumer,
            kafka_producer=MockKafkaProducer(mock_result),
            filter_handler=init_filter_handler(mock_opr_config, "test_pipeline"),
            output_topic="test_topic",
            pipeline_id="test_pipeline",
            operator_id="test_operator"
        )
        while not mock_kafka_consumer.empty():
            mock_operator._OperatorBase__route()

    def test_run(self):
        try:
            with open("tests/resources/opr_config.json") as file:
                opr_config = json.load(file)
            operator = algo.Operator(energy_src_id="device:pv:1", weather_src_id="weather_import")
            operator.init(
                kafka_consumer=None,
                kafka_producer=None,
                filter_handler=init_filter_handler(opr_config, None),
                output_topic=None,
                pipeline_id=None,
                operator_id=None
            )
            with open("tests/resources/messages.txt") as file:
                for line in file:
                    results = operator._OperatorBase__call_run(json.loads(line.strip()))
                    for result in results:
                        print(result)
        except FileNotFoundError as ex:
            self.skipTest(ex)


if __name__ == '__main__':
    unittest.main()
