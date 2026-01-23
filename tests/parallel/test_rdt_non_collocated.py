"""RDT behavior check for non-collocated Ray actors."""

from __future__ import annotations

from pathlib import Path

import pytest
import ray
import torch
from ray.experimental.collective import create_collective_group
from ray.exceptions import GetTimeoutError
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.placement_group import placement_group

pytestmark = [pytest.mark.gpu]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = Path(__file__).resolve().parent


@ray.remote(enable_tensor_transport=True, num_gpus=1, num_cpus=1)
class RdtProducer:
    def get_assigned_gpu_id(self) -> int:
        return int(ray.get_gpu_ids()[0])

    @ray.method(tensor_transport="nccl")
    def produce(self, size: int = 4) -> torch.Tensor:
        return torch.arange(size, device="cuda", dtype=torch.float32)


@ray.remote(enable_tensor_transport=True, num_gpus=1, num_cpus=1)
class RdtConsumer:
    def get_assigned_gpu_id(self) -> int:
        return int(ray.get_gpu_ids()[0])

    def consume(self, tensor: torch.Tensor) -> dict[str, int | bool]:
        return {
            "is_cuda": tensor.is_cuda,
            "assigned_gpu_id": int(ray.get_gpu_ids()[0]),
        }


def test_rdt_non_collocated_tensor_transfer():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for RDT non-collocated tensor transfer test.")
    if torch.cuda.device_count() < 2:
        pytest.skip("RDT non-collocated test requires at least 2 GPUs.")

    ray.init(
        address="auto",
        ignore_reinit_error=True,
        include_dashboard=False,
        runtime_env={
            "working_dir": str(PROJECT_ROOT),
            "py_modules": [str(TESTS_ROOT)],
        },
    )
    try:
        pg = placement_group([{"CPU": 1, "GPU": 1}, {"CPU": 1, "GPU": 1}], strategy="PACK")
        try:
            ray.get(pg.ready(), timeout=30)
        except GetTimeoutError:
            pytest.skip("Timed out waiting for a 2-GPU placement group; skipping cross-GPU RDT test.")

        producer = RdtProducer.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=0,
                placement_group_capture_child_tasks=True,
            )
        ).remote()
        consumer = RdtConsumer.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=1,
                placement_group_capture_child_tasks=True,
            )
        ).remote()

        producer_device, consumer_device = ray.get(
            [producer.get_assigned_gpu_id.remote(), consumer.get_assigned_gpu_id.remote()]
        )
        if producer_device == consumer_device:
            pytest.skip("Actors are collocated on the same GPU; need cross-GPU transfer.")

        create_collective_group([producer, consumer], backend="nccl")

        tensor_ref = producer.produce.remote()
        consumer_result = ray.get(consumer.consume.remote(tensor_ref))
        assert consumer_result["is_cuda"], "Tensor should stay on GPU when transferred via RDT."
        assert (
            consumer_result["assigned_gpu_id"] == consumer_device
        ), "Consumer should keep its assigned GPU id while receiving the tensor."

        driver_tensor = ray.get(tensor_ref, _tensor_transport="object_store")
        assert isinstance(driver_tensor, torch.Tensor)
    finally:
        ray.shutdown()
