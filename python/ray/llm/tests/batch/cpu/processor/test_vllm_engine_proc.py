import ray
import asyncio
import uuid
from unittest.mock import patch
from typing import List, Dict, Any, AsyncIterator, Type

from ray.llm._internal.batch.processor.vllm_engine_proc import build_vllm_engine_processor, vLLMEngineProcessorConfig
from ray.llm._internal.batch.stages.vllm_engine_stage import vLLMOutputData
from ray.llm._internal.batch.stages import StatefulStage
from ray.llm._internal.batch.stages.base import StatefulStageUDF


class MockVLLMEngineUDF(StatefulStageUDF):
    def __init__(
        self,
        data_column: str,
        expected_input_keys: List[str],
        *args,
        **kwargs,
    ):
        super().__init__(data_column, expected_input_keys)
        self._num_running_batches = 0
        self._num_running_rows = 0

    async def udf(
        self, batch: List[Dict[str, Any]]
    ) -> AsyncIterator[Dict[str, Any]]:
        self._num_running_batches += 1
        self._num_running_rows += len(batch)
        batch_uuid = uuid.uuid4()
        # mimic llm processing time
        await asyncio.sleep(8)
        idx = 0
        for row in batch:
            output = vLLMOutputData(
                prompt=row["prompt"],
                prompt_token_ids=None,
                num_input_tokens=8,
                generated_tokens=[idx+3,idx+4,idx+5],
                generated_text="fake generated text",
            ).model_dump()
            yield {
                self.IDX_IN_BATCH_COLUMN: row[self.IDX_IN_BATCH_COLUMN],
                **output,
                "request_id": f"fake-request-id-{batch_uuid.hex}-{idx}",
                "batch_uuid": batch_uuid.hex,
                "time_taken_llm": 0.001,
                "params": {},
                "num_running_batches": self._num_running_batches,
                "num_running_rows": self._num_running_rows,
            }
            idx += 1
        self._num_running_batches -= 1
        self._num_running_rows -= len(batch)

class MockVLLMEngineStage(StatefulStage):
    fn: Type[StatefulStageUDF] = MockVLLMEngineUDF

def preprocess(row: dict[str, Any]) -> dict[str, Any]:
    return dict(
        messages=[
            {
                "role": "user",
                "content": f"2 ** {row['id']} = ?",
            },
        ],
    )

def postprocess(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
    }

class TestVLLMEngineProcessor:
    def test_vllm_engine_saturated_concurrency(self,
    # cpu-only assets defined in conftest.py, needed to initialize tokenizer etc.
    model_pixtral_12b):
        num_rows_to_process = 1024
        with patch("ray.llm._internal.batch.processor.vllm_engine_proc.vLLMEngineStage", side_effect=MockVLLMEngineStage):
            config = vLLMEngineProcessorConfig(
                model_source=model_pixtral_12b,
                engine_kwargs={},
                batch_size=8,
                max_concurrent_batches=32,
                experimental=dict(
                    max_tasks_in_flight_per_actor=64,
                ),
            )
            processor = build_vllm_engine_processor(config,
                preprocess=preprocess,
                postprocess=postprocess,
            )
            output = processor(ray.data.range(num_rows_to_process)).materialize()
            print(f"{output.max('num_running_batches')=}")
            print(f"{output.max('num_running_rows')=}")
