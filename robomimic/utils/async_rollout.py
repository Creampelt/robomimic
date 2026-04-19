"""
Background-thread rollout manager for BC training.

The training loop normally stops to run evaluation rollouts every
``experiment.rollout.rate`` epochs. With a warp env + long horizon, this
can be the biggest chunk of wall time in a run. ``AsyncRolloutManager``
moves the rollouts to a worker thread so training keeps making progress
while evaluation runs in parallel on the same GPU (Python threads hold
the GIL but CUDA kernel launches are async, so compute overlaps cleanly).

Design summary (see CLAUDE.md for full context):
  - The manager owns the rollout envs *and* a shadow copy of the algo
    (``copy.deepcopy`` once at init). All env / policy work happens on
    the worker thread; the main thread never touches either after init.
  - Each ``submit(epoch)`` call eagerly clones the main model's weights
    (``detach().clone()`` per tensor, stays on the same device) and
    pushes ``(epoch, state_dict_snapshot)`` onto a bounded queue. If the
    queue is full, submit blocks — this is the intended backpressure so
    training can't get arbitrarily far ahead of evaluation.
  - Completed rollouts are pushed onto a result queue. The main thread
    calls ``drain()`` once per training iter to collect results and log
    them via wandb's ``local_step`` metric (past epoch numbers). At
    shutdown, ``drain_blocking()`` waits for remaining rollouts.

Only BC (train-on-fixed-dataset) is supported — the main thread does not
step the envs, so no lock is needed. If online data collection gets
added, a mutex around env access would need to follow.
"""

import copy
import os
import queue
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import torch

import robomimic.utils.train_utils as TrainUtils
from robomimic.algo import RolloutPolicy


# Sentinel placed on the request queue to tell the worker to exit.
_SHUTDOWN = object()


@dataclass
class RolloutResult:
    """One completed async rollout, ready to log against ``local_step=epoch``."""

    epoch: int
    all_rollout_logs: "OrderedDict[str, dict]"
    video_paths: "OrderedDict[str, str]"
    wall_time: float
    error: BaseException | None = None
    # The exact weights that produced this rollout (cloned off the main
    # thread at submit time, kept alive through the worker). Main thread
    # consumes this to write a best-score checkpoint with the weights
    # that actually scored the metric, not the current live weights
    # which have already moved on.
    nets_state_dict: dict | None = None
    # per-env-key best-metric comparisons get run on the main thread, so we
    # carry whatever ``should_save_from_rollout_logs`` needs to consume
    # directly in ``all_rollout_logs``.
    extra: dict = field(default_factory=dict)


class AsyncRolloutManager:
    """Background-thread wrapper around ``TrainUtils.rollout_with_stats``.

    Only used for evaluation rollouts during BC training. Training never
    touches the envs or the shadow model — all env + policy work happens
    on the single worker thread.
    """

    def __init__(
        self,
        envs: "OrderedDict",
        model,
        *,
        horizon: int,
        num_episodes: int,
        use_warp: bool,
        use_goals: bool,
        video_dir: str | None,
        video_skip: int,
        terminate_on_success: bool,
        queue_size: int = 2,
    ):
        self._envs = envs
        self._horizon = horizon
        self._num_episodes = num_episodes
        self._use_warp = use_warp
        self._use_goals = use_goals
        self._video_dir = video_dir
        self._video_skip = video_skip
        self._terminate_on_success = terminate_on_success

        # Shadow algo: deep-copied once. Weights get overwritten via
        # load_state_dict on every submit; optimizer / lr_scheduler state
        # is irrelevant for rollout but stays around as a (tiny) cost of
        # using deepcopy vs a hand-rebuilt algo.
        self._shadow_model = copy.deepcopy(model)
        self._obs_normalization_stats = getattr(model, "obs_normalization_stats", None)
        self._action_normalization_stats = getattr(model, "action_normalization_stats", None)

        self._request_q: "queue.Queue" = queue.Queue(maxsize=max(1, queue_size))
        self._result_q: "queue.Queue" = queue.Queue()
        self._inflight = 0
        self._inflight_lock = threading.Lock()
        self._stopped = False

        self._worker = threading.Thread(target=self._run, name="async-rollout", daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------
    # public API (main thread)
    # ------------------------------------------------------------------

    def submit(
        self,
        epoch: int,
        model,
        obs_normalization_stats=None,
        action_normalization_stats=None,
    ) -> None:
        """Clone weights off the main thread and queue a rollout for ``epoch``.

        Blocks if the request queue is at capacity (intentional backpressure).
        """
        if self._stopped:
            raise RuntimeError("AsyncRolloutManager is shut down")

        # Per-tensor detach+clone: keeps tensors on their original device
        # (cheap GPU→GPU copy) but breaks storage-sharing with the live
        # model, so training can continue mutating weights while the
        # worker consumes the snapshot.
        state_dict = {k: v.detach().clone() for k, v in model.nets.state_dict().items()}

        # Normalization stats are numpy arrays that don't change across
        # epochs for BC; pick up any fresh reference in case a caller
        # re-computes them mid-training.
        if obs_normalization_stats is not None:
            self._obs_normalization_stats = obs_normalization_stats
        if action_normalization_stats is not None:
            self._action_normalization_stats = action_normalization_stats

        with self._inflight_lock:
            self._inflight += 1
        self._request_q.put((epoch, state_dict))

    def drain(self) -> list[RolloutResult]:
        """Return every result currently available (non-blocking)."""
        results: list[RolloutResult] = []
        while True:
            try:
                results.append(self._result_q.get_nowait())
            except queue.Empty:
                break
        return results

    def drain_blocking(self, timeout: float | None = None) -> list[RolloutResult]:
        """Wait for all in-flight rollouts to finish, then drain.

        Polls the in-flight counter with a small sleep rather than
        blocking on the result queue, so multiple completions can finish
        before we return. ``timeout=None`` waits indefinitely.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._inflight_lock:
                if self._inflight == 0 and self._request_q.empty():
                    break
            if deadline is not None and time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        return self.drain()

    def pending_count(self) -> int:
        """Number of submitted rollouts that haven't been drained yet."""
        with self._inflight_lock:
            return self._inflight

    def close(self, timeout: float = 10.0) -> None:
        """Stop the worker. Safe to call multiple times."""
        if self._stopped:
            return
        self._stopped = True
        try:
            self._request_q.put_nowait(_SHUTDOWN)
        except queue.Full:
            # worker will see shutdown after consuming the backlog; push
            # again once there's room
            self._request_q.put(_SHUTDOWN)
        self._worker.join(timeout=timeout)

    # ------------------------------------------------------------------
    # worker thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while True:
            item = self._request_q.get()
            if item is _SHUTDOWN:
                return
            epoch, state_dict = item
            start = time.time()
            error: BaseException | None = None
            all_rollout_logs: OrderedDict = OrderedDict()
            video_paths: OrderedDict = OrderedDict()
            try:
                # Restore weights onto the shadow model. no_grad so we
                # don't touch autograd state on the shadow.
                with torch.no_grad():
                    self._shadow_model.nets.load_state_dict(state_dict)
                policy = RolloutPolicy(
                    self._shadow_model,
                    obs_normalization_stats=self._obs_normalization_stats,
                    action_normalization_stats=self._action_normalization_stats,
                    use_warp=self._use_warp,
                )
                all_rollout_logs, video_paths = TrainUtils.rollout_with_stats(
                    policy=policy,
                    envs=self._envs,
                    horizon=self._horizon,
                    use_goals=self._use_goals,
                    num_episodes=self._num_episodes,
                    render=False,
                    video_dir=self._video_dir,
                    epoch=epoch,
                    video_skip=self._video_skip,
                    terminate_on_success=self._terminate_on_success,
                )
            except BaseException as e:  # surface the error to the main thread
                error = e
            finally:
                with self._inflight_lock:
                    self._inflight -= 1
                # Pass the snapshot through so the main thread can save
                # exact-weight best-score checkpoints. ``load_state_dict``
                # only reads from the dict; the tensors are still good
                # clones detached from the live model.
                self._result_q.put(
                    RolloutResult(
                        epoch=epoch,
                        all_rollout_logs=all_rollout_logs,
                        video_paths=video_paths,
                        wall_time=time.time() - start,
                        error=error,
                        nets_state_dict=state_dict if error is None else None,
                    )
                )
