"""
Tensor transfer utilities for Ray actors.

Supports two modes of tensor transfer:
1. CUDA IPC: Zero-copy tensor sharing for processes on the same GPU
2. Ray Direct Transport: For processes on different GPUs
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import ray
import torch

from .utils import get_physical_gpu_id

logger = logging.getLogger(__name__)


def create_ipc_handle(tensor: torch.Tensor) -> Tuple[Any, str, Any]:
    """Create a CUDA IPC handle for a tensor with synchronization event.

    Args:
        tensor: PyTorch tensor to create IPC handle for (must be on GPU)

    Returns:
        Tuple of (ipc_handle, physical_gpu_id, event_ipc_handle)
    """
    from torch.multiprocessing.reductions import reduce_tensor

    if not tensor.is_cuda:
        raise ValueError("Tensor must be on CUDA device to create IPC handle")

    # Ensure tensor is contiguous
    tensor = tensor.contiguous()

    # Create an interprocess-shareable CUDA event
    # This event can be shared across processes via IPC
    event = torch.cuda.Event(interprocess=True)

    # Record the event on the current stream
    # This marks the point when all prior operations (including tensor computation) are complete
    torch.cuda.current_stream().record_event(event)

    # Create IPC handle for the tensor
    ipc_handle = reduce_tensor(tensor)
    physical_gpu_id = get_physical_gpu_id()

    # Get IPC handle for the event
    event_ipc_handle = event.ipc_handle()

    return ipc_handle, physical_gpu_id, event_ipc_handle


def reconstruct_tensor_from_ipc(
    ipc_handle: Any,
    physical_gpu_id: str,
    sender_gpu_id: str,
    event_ipc_handle: Optional[Any] = None,
) -> torch.Tensor:
    """Reconstruct a tensor from a CUDA IPC handle with synchronization.

    Args:
        ipc_handle: The IPC handle from reduce_tensor
        physical_gpu_id: Physical GPU ID of the receiving process
        sender_gpu_id: Physical GPU ID of the sending process
        event_ipc_handle: Optional IPC handle for synchronization event

    Returns:
        Reconstructed tensor (zero-copy reference)
    """
    if physical_gpu_id != sender_gpu_id:
        raise ValueError(
            f"IPC handle is for GPU {sender_gpu_id}, but receiver is on GPU {physical_gpu_id}. "
            "CUDA IPC only works for processes on the same physical GPU."
        )

    # Get the device index (accounting for CUDA_VISIBLE_DEVICES)
    device_id = torch.cuda.current_device()

    # If event handle is provided, wait for sender's event before using tensor
    if event_ipc_handle is not None:
        # Reconstruct the event from IPC handle
        event_remote = torch.cuda.Event.from_ipc_handle(device=device_id, handle=event_ipc_handle)

        # Make current stream wait for the sender's event
        # This ensures sender's computation is complete before we use the tensor
        # This is asynchronous - doesn't block CPU, only GPU stream
        torch.cuda.current_stream().wait_event(event_remote)

    # Reconstruct the tensor
    func, args = ipc_handle
    list_args = list(args)
    # Update device ID to match current process's device mapping
    list_args[6] = device_id
    tensor = func(*list_args)

    return tensor


class TensorTransferRequest:
    """Container for tensor transfer metadata."""

    def __init__(
        self,
        tensor: Optional[torch.Tensor] = None,
        ipc_handle: Optional[Any] = None,
        sender_gpu_id: Optional[str] = None,
        shape: Optional[Tuple] = None,
        dtype: Optional[torch.dtype] = None,
        use_ipc: bool = False,
        event_ipc_handle: Optional[Any] = None,
    ):
        """Initialize transfer request.

        Args:
            tensor: The actual tensor (for Ray Direct Transport)
            ipc_handle: CUDA IPC handle (for same-GPU transfer)
            sender_gpu_id: Physical GPU ID of sender
            shape: Tensor shape
            dtype: Tensor dtype
            use_ipc: Whether this transfer uses CUDA IPC
            event_ipc_handle: CUDA event IPC handle for synchronization (IPC only)
        """
        self.tensor = tensor
        self.ipc_handle = ipc_handle
        self.sender_gpu_id = sender_gpu_id
        self.shape = shape
        self.dtype = dtype
        self.use_ipc = use_ipc
        self.event_ipc_handle = event_ipc_handle

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Ray object store."""
        return {
            "tensor": self.tensor,
            "ipc_handle": self.ipc_handle,
            "sender_gpu_id": self.sender_gpu_id,
            "shape": self.shape,
            "dtype": str(self.dtype) if self.dtype else None,
            "use_ipc": self.use_ipc,
            "event_ipc_handle": self.event_ipc_handle,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TensorTransferRequest":
        """Create from dictionary."""
        dtype_str = data.get("dtype")
        dtype = getattr(torch, dtype_str.split(".")[-1]) if dtype_str else None

        return cls(
            tensor=data.get("tensor"),
            ipc_handle=data.get("ipc_handle"),
            sender_gpu_id=data.get("sender_gpu_id"),
            shape=data.get("shape"),
            dtype=dtype,
            use_ipc=data.get("use_ipc", False),
            event_ipc_handle=data.get("event_ipc_handle"),
        )


def prepare_tensor_for_transfer(
    tensor: torch.Tensor,
    receiver_gpu_ids: List[str],
    sender_gpu_id: str,
    use_ipc_if_same_gpu: bool = True,
) -> TensorTransferRequest:
    """Prepare a tensor for transfer to receiver(s).

    Automatically chooses between CUDA IPC (same GPU) and Ray Direct Transport (different GPUs).

    Args:
        tensor: Tensor to transfer
        receiver_gpu_ids: List of physical GPU IDs of receivers
        sender_gpu_id: Physical GPU ID of sender
        use_ipc_if_same_gpu: Whether to use CUDA IPC for same-GPU transfers

    Returns:
        TensorTransferRequest object
    """
    # Check if all receivers are on the same GPU as sender
    all_same_gpu = all(gpu_id == sender_gpu_id for gpu_id in receiver_gpu_ids)
    use_ipc = all_same_gpu and use_ipc_if_same_gpu and tensor.is_cuda

    if use_ipc:
        logger.debug(f"Using CUDA IPC for same-GPU transfer (GPU: {sender_gpu_id})")
        ipc_handle, gpu_id, event_ipc_handle = create_ipc_handle(tensor)
        return TensorTransferRequest(
            ipc_handle=ipc_handle,
            sender_gpu_id=gpu_id,
            shape=tensor.shape,
            dtype=tensor.dtype,
            use_ipc=True,
            event_ipc_handle=event_ipc_handle,
        )
    else:
        logger.debug("Using Ray Direct Transport for cross-GPU transfer")
        return TensorTransferRequest(
            tensor=tensor,
            sender_gpu_id=sender_gpu_id,
            shape=tensor.shape,
            dtype=tensor.dtype,
            use_ipc=False,
        )


def receive_tensor(
    transfer_request: TensorTransferRequest,
    receiver_gpu_id: str,
) -> torch.Tensor:
    """Receive a tensor from a transfer request.

    Args:
        transfer_request: The transfer request from sender
        receiver_gpu_id: Physical GPU ID of receiver

    Returns:
        The received tensor
    """
    if transfer_request.use_ipc:
        logger.debug("Receiving tensor via CUDA IPC")
        return reconstruct_tensor_from_ipc(
            transfer_request.ipc_handle,
            receiver_gpu_id,
            transfer_request.sender_gpu_id,
            transfer_request.event_ipc_handle,
        )
    else:
        logger.debug("Receiving tensor via Ray Direct Transport")
        return transfer_request.tensor


def gather_gpu_ids(actors: List[ray.actor.ActorHandle]) -> Dict[int, str]:
    """Gather physical GPU IDs from all actors.

    Args:
        actors: List of Ray actor handles

    Returns:
        Dict mapping actor index to physical GPU ID
    """
    gpu_id_refs = [actor.get_physical_gpu_id.remote() for actor in actors]
    gpu_ids = ray.get(gpu_id_refs)
    return {i: gpu_id for i, gpu_id in enumerate(gpu_ids)}


def detect_collocated_actors(
    sender_actors: List[ray.actor.ActorHandle],
    receiver_actors: List[ray.actor.ActorHandle],
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """Detect which sender-receiver pairs are collocated on the same GPU.

    Args:
        sender_actors: List of sender actor handles
        receiver_actors: List of receiver actor handles

    Returns:
        Tuple of (sender_to_receivers, receiver_to_senders) mappings
        where each maps actor index to list of collocated actor indices
    """
    sender_gpu_ids = gather_gpu_ids(sender_actors)
    receiver_gpu_ids = gather_gpu_ids(receiver_actors)

    logger.info(f"Sender GPU IDs: {sender_gpu_ids}")
    logger.info(f"Receiver GPU IDs: {receiver_gpu_ids}")

    # Build collocation maps
    sender_to_receivers = {}
    receiver_to_senders = {}

    for s_idx, s_gpu in sender_gpu_ids.items():
        collocated_receivers = [r_idx for r_idx, r_gpu in receiver_gpu_ids.items() if r_gpu == s_gpu]
        sender_to_receivers[s_idx] = collocated_receivers

    for r_idx, r_gpu in receiver_gpu_ids.items():
        collocated_senders = [s_idx for s_idx, s_gpu in sender_gpu_ids.items() if s_gpu == r_gpu]
        receiver_to_senders[r_idx] = collocated_senders

    logger.info(f"Collocated sender->receiver mapping: {sender_to_receivers}")
    logger.info(f"Collocated receiver->sender mapping: {receiver_to_senders}")

    return sender_to_receivers, receiver_to_senders
