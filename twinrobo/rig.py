"""The standard way to use any twin in a simulator.

Every twin (`CameraTwin`, `StereoTwin`, `LidarTwin`, `DeviceTwin`) has the same steps:

    twin = StereoTwin.from_catalog("stereolabs/zed-x/2.2mm")
    rig = twin.attach(backend, camera="wrist")   # bind it to a simulator
    obs = rig.capture()                          # one `Observation`
    obs["left"].rgb
    rig.reset()                                  # a new episode

A backend is what a simulator gives TwinRobo: `twinrobo.plugins.mujoco.MujocoRenderer`,
`twinrobo.plugins.mujoco.RobosuiteRenderer` or `twinrobo.plugins.isaac.IsaacBackend`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any


class Observation(Mapping):
    """What one capture returned, by stream name.

    The streams keep their own types: `CameraFrame` for cameras and `LidarFrame` for lidars.
    A camera's stream is ``"camera"``, a stereo pair's are ``"left"`` and ``"right"``, a lidar's
    is ``"lidar"``. A `DeviceTwin` names them by part: ``"rgb"``, ``"ir/left"``.
    """

    def __init__(self, streams: dict[str, Any], timestamp: float | None = None):
        self.streams = dict(streams)
        self.timestamp = timestamp

    def __getitem__(self, name: str) -> Any:
        return self.streams[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.streams)

    def __len__(self) -> int:
        return len(self.streams)

    def __repr__(self) -> str:
        return f"Observation({list(self.streams)}, timestamp={self.timestamp})"


class Rig:
    """A twin attached to a simulator. Made by a twin's ``attach``."""

    def capture(
        self, timestamp: float | None = None, same_frame: bool = False, **options: Any
    ) -> Observation:
        """Capture one observation of the simulator's current state. Physics does not advance.

        Args:
            timestamp: Stored on the observation and on camera frames.
            same_frame: This rig shares a simulator frame another rig already rendered.
            **options: Passed to the capture, such as ``force_depth=True`` for a camera.

        Returns:
            The `Observation`.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Start again, as at a new episode: clears exposure, ISP state and frame counters."""
        raise NotImplementedError

    def close(self) -> None:
        """Remove what the rig added to the simulator."""


class CameraRig(Rig):
    """One camera in a simulator. ``adapter`` has the simulator-specific camera."""

    def __init__(self, adapter: Any, backend: Any):
        self.adapter = adapter
        self.backend = backend

    def capture(
        self, timestamp: float | None = None, same_frame: bool = False, **options: Any
    ) -> Observation:
        options = {**self.backend.capture_options, **options}
        if same_frame:
            options |= self.backend.same_frame  # one observation is one simulator frame
        frame = self.adapter.get_frame(timestamp=timestamp, **options)
        return Observation({"camera": frame}, timestamp)

    def reset(self) -> None:
        self.adapter.reset()

    def close(self) -> None:
        close = getattr(self.adapter, "close", None)
        if close is not None:
            close()


class LidarRig(Rig):
    """One lidar in a simulator. Each capture is the next frame of its noise sequence.

    ``make`` builds the simulator's lidar. It is built again when the simulator replaces its
    model, as robosuite may on reset.
    """

    def __init__(self, make: Callable[[], Any], backend: Any):
        self.make = make
        self.backend = backend
        self.adapter = make()
        self.frame = 0

    def capture(
        self, timestamp: float | None = None, same_frame: bool = False, **options: Any
    ) -> Observation:
        if (
            self.backend.model is not self.adapter.model
            or self.backend.data is not self.adapter.data
        ):
            self.adapter = self.make()
        scan = self.adapter.scan(frame=self.frame, **options)
        self.frame += 1
        return Observation({"lidar": scan}, timestamp)

    def reset(self) -> None:
        self.frame = 0


class GroupRig(Rig):
    """Several rigs captured together from one simulator frame.

    ``capture`` passes an option named after a part to that part only (``left={...}``), and
    any other option to every part.
    """

    def __init__(self, parts: dict[str, Rig]):
        self.parts = parts

    def capture(
        self, timestamp: float | None = None, same_frame: bool = False, **options: Any
    ) -> Observation:
        own = {name: options.pop(name) for name in self.parts if name in options}
        streams: dict[str, Any] = {}
        for name, rig in self.parts.items():
            part = rig.capture(timestamp, same_frame, **(options | own.get(name, {})))
            same_frame = True
            single = len(part) == 1
            streams |= {name if single else f"{name}/{k}": v for k, v in part.items()}
        return Observation(streams, timestamp)

    def reset(self) -> None:
        for rig in self.parts.values():
            rig.reset()

    def close(self) -> None:
        for rig in self.parts.values():
            rig.close()
