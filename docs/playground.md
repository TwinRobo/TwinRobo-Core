---
hide:
  - navigation
  - toc
nav_icon: material/robot-industrial
---

<h1 class="tr-sr-only">Try in Simulator</h1>

<div class="tr-demo" markdown="0"><iframe src="https://twinrobo.github.io/TwinRobo-Preview/" title="TwinRobo Preview" loading="lazy" allow="fullscreen"></iframe></div>
<p class="tr-demo-cap"><a href="https://twinrobo.github.io/TwinRobo-Preview/" target="_blank" rel="noopener">Open full screen</a> · <a href="https://github.com/TwinRobo/TwinRobo-Preview" target="_blank" rel="noopener">TwinRobo-Preview</a> (precomputed with TwinRobo Studio; runs in the browser)</p>

## Supported simulators

Every catalog camera works in every simulator below, with every rendering
method: **PSF** (the simulator's pinhole render, blurred and distorted by the
lens; fastest) and **ray cast** (every pixel traced through the real lens;
occlusion-aware blur, the lens' own distortion).

| Simulator | What you get | Environment | Methods |
|---|---|---|---|
| MuJoCo | any MJCF model, `mujoco.Renderer` or robosuite | `pip install -e ".[libero]"` | PSF, ray cast |
| LIBERO | LIBERO tasks and demos, policies unchanged | robosuite 1.4 | PSF, ray cast |
| RoboCasa | procedural kitchens, human demos replayed exactly | robosuite 1.5 (own environment) | PSF, ray cast |
| Isaac Sim 6.x | any USD stage, RTX rendering | Docker image in `docker/isaac/` | PSF, ray cast |

Each adapter renders the simulator's camera at the twin's resolution and field
of view, from the same pose, and returns a `CameraFrame` (`frame.rgb`, linear, on
the GPU). Pass `render="raycast"` to trace the real lens.

=== "MuJoCo"

    ```python
    import mujoco
    from twinrobo import CameraTwin
    from twinrobo.plugins.mujoco import MujocoCameraTwin, MujocoRenderer

    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)
    twin = CameraTwin.from_catalog("stereolabs/zed-x/2.2mm")
    cam = MujocoCameraTwin(twin, MujocoRenderer(model, data), camera="wrist")
    frame = cam.get_frame()  # render="raycast" to trace every pixel through the lens
    ```

    Inside robosuite, use `RobosuiteRenderer(env)` as the backend (a second
    `mujoco.Renderer` in a robosuite process corrupts its renders).
    [More](simulators.md#mujoco)

=== "LIBERO"

    ```python
    from twinrobo import CameraTwin
    from twinrobo.datasets.libero import CameraTwinLiberoEnv, make_env

    env, task, init_states = make_env("libero_spatial", 0, resolution=128)
    env = CameraTwinLiberoEnv(env, {"agentview": CameraTwin.from_catalog("stereolabs/zed-x/2.2mm")})
    env.seed(0)
    env.reset()
    obs = env.set_init_state(init_states[0])  # obs["agentview_image"] comes from the twin
    ```

    Observations keep their shape, dtype and image convention, so eval scripts
    and policies run unchanged. Needs `LIBERO_ROOT` and `LIBERO_CONFIG_PATH`.
    [More](simulators.md#libero)

=== "RoboCasa"

    ```python
    from twinrobo.datasets.robocasa import env_args, make_env, read_episode, reset_to_episode, setup_robocasa

    setup_robocasa()  # checks robosuite 1.5, links $ROBOCASA_ASSETS
    env = make_env(env_args(dataset_dir))  # a LeRobot dataset dir
    states, kitchen_xml, ep_meta = read_episode(dataset_dir, "episode_000000")
    reset_to_episode(env, kitchen_xml, ep_meta)  # the episode's own kitchen
    env.sim.set_state_from_flattened(states[t])  # then render any camera at frame t
    ```

    RoboCasa needs robosuite 1.5 (LIBERO needs 1.4), so it gets its own
    environment. [More](simulators.md#robocasa)

=== "Isaac Sim"

    ```python
    from twinrobo import CameraTwin
    from twinrobo.plugins.isaac import IsaacCameraTwin

    twin = CameraTwin.from_catalog("stereolabs/zed-x/2.2mm")
    cam = IsaacCameraTwin(twin, "/World/Camera")  # any UsdGeom.Camera prim
    frame = cam.get_frame()                       # render="raycast" for ray casting
    cam.close()                                   # removes what it added to the stage
    ```

    ```bash
    docker/isaac/run.sh examples/01_isaac_camera.py   # Isaac Sim 6.1 in Docker
    ```

    Your camera is not changed: the adapter adds a child camera with the twin's
    intrinsics. [More](simulators.md#isaac-sim)

Another simulator? Any renderer that gives RGB and depth can use
`twin.process(rgb, depth)`; a full adapter (for the lens-ray methods) is three
methods. See [contributing](contributing.md).
