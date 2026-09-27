# Simulators

| Simulator | Scenes and data | Status |
|---|---|---|
| MuJoCo | any MJCF model; `mujoco.Renderer` or robosuite backends | supported |
| LIBERO (robosuite 1.4) | LIBERO tasks and HDF5 demo datasets, 9 robot options | supported |
| RoboCasa (robosuite 1.5) | procedurally generated kitchens, LeRobot demo datasets | supported |
| Isaac Sim 6.x | USD scenes | planned (the adapter is a stub) |

## MuJoCo

`twinrobo.mujoco` renders a MuJoCo camera at the twin's resolution and
paraxial FoV, from the same camera pose, with metric depth, and runs the
TwinRobo pipeline. `MujocoCameraTwin(twin, backend, camera=...)` works with
`MujocoRenderer(model, data)` for plain MuJoCo and `RobosuiteRenderer(env)`
inside robosuite (a second `mujoco.Renderer` in a robosuite process corrupts
robosuite's renders).

## LIBERO

```bash
pip install -e ".[libero]"
export LIBERO_ROOT=/path/to/LIBERO                 # checkout containing libero/libero
export LIBERO_CONFIG_PATH=/dir/with/libero/config  # avoids LIBERO's stdin prompt
export MUJOCO_GL=egl
```

```python
from twinrobo import CameraTwin
from twinrobo.libero import CameraTwinLiberoEnv, make_env

env, task, init_states = make_env("libero_spatial", 0, resolution=128)
twin = CameraTwin.from_catalog("examples/cellphone80deg")
env = CameraTwinLiberoEnv(env, {"agentview": twin})
env.seed(0)
env.reset()
obs = env.set_init_state(init_states[0])  # obs["agentview_image"] comes from the twin
```

- **Online:** `CameraTwinLiberoEnv` replaces `obs[f"{camera}_image"]`, keeping
  its shape, dtype and image convention, so eval scripts and policies run
  unchanged.
- **Replay:** demos replay from their recorded states
  (`twinrobo.libero.replay`), so any camera can be placed in a recorded
  episode.
- **Geometry:** the twin's frame is center-cropped to the observation's aspect
  ratio and area-resized. `match_fov=False` keeps LIBERO's FoV and applies only
  the optics.
- **Robots:** besides the Franka Panda, robosuite's Sawyer, UR5e, KUKA iiwa,
  Kinova Jaco and Kinova Gen3 are registered for LIBERO scenes
  (`twinrobo.libero.robots`, LIBERO unmodified). Multi-camera variants of the
  Panda, UR5e and Gen3 carry a ZED X Mini stereo module at the wrist and an
  over-the-shoulder camera; cameras add no joints, so a variant replays its
  base arm's demos.
- **Exact replay:** LIBERO randomizes the poses of its jointless fixtures
  (cabinet, stove, ...) per episode, outside the recorded state. TwinRobo
  applies each episode's fixture poses from its model XML, which raised
  replay-vs-stored-image PSNR from about 26 to about 35 dB.

Measured on `libero_spatial` task 0, RTX 3090, example spec at 1920x1200:

| | |
|---|---|
| Step time (wrapped, agentview only) | about 300-340 ms (render ~15 ms, optics the rest) |
| Identity twin at LIBERO FoV/res vs original obs | exact (at most 1/255, sRGB round trip) |
| Optics effect at 128 / 256 / 512 px (ideal vs twin, same FoV) | 43 / 39.6 / 36.5 dB |

At LIBERO's 128 px, a sharp lens's 2-3 px blur at 1920x1200 is mostly
downsampled away and the camera's geometry dominates. Optics matter at higher
policy resolutions, with blurrier lenses or close focus.

## RoboCasa

RoboCasa brings realistic scenes: procedurally assembled kitchens with textured
fixtures, appliances and Objaverse objects, and human demos of the PandaOmron
mobile manipulator. Each episode stores its generated kitchen
(`model.xml.gz`) and its simulator states, so it is replayed exactly: 35-37 dB
against the recorded (H.264) videos.

RoboCasa needs robosuite 1.5 and LIBERO needs 1.4.1, so each gets its own
environment:

```bash
conda create -n twinrobo-robocasa python=3.12 -y
conda activate twinrobo-robocasa
pip install -e ".[dev]"
pip install "numpy==2.2.5" "numba==0.61.2" "scipy==1.15.3" "mujoco==3.3.1" \
    warp-lang pyarrow h5py matplotlib "imageio[ffmpeg]"
pip install --no-deps "robosuite @ git+https://github.com/ARISE-Initiative/robosuite@5ce6643f" \
    "robocasa @ git+https://github.com/robocasa/robocasa@4f8a2980"
export ROBOCASA_ASSETS=<robocasa asset pack>   # objects/, textures/, fixtures/, ...
```

The asset pack is linked into the installed package; the pack is not modified.

```python
from twinrobo.robocasa import env_args, make_env, read_episode, reset_to_episode, setup_robocasa

setup_robocasa()  # checks robosuite 1.5, links $ROBOCASA_ASSETS
env = make_env(env_args(dataset_dir))  # a LeRobot dataset dir (meta/, extras/)
states, kitchen_xml, ep_meta = read_episode(dataset_dir, "episode_000000")
reset_to_episode(env, kitchen_xml, ep_meta)  # the episode's own kitchen
env.sim.set_state_from_flattened(states[t])  # then render any camera at frame t
```

- Loading an episode takes about 10 s: every episode has its own kitchen.
  Frames within an episode are fast.
- Ray cast costs about 2.2 s per 1920x1200 frame in a kitchen (570k
  triangles); pupil raster costs the same as in LIBERO.
- Robot: PandaOmron, the robot the demos were recorded with.

## Known constraints

- DeepLens requires Python 3.12 and pins `torch`. Isaac Sim 6.x also targets
  Python 3.12; older Isaac releases cannot import DeepLens. The torch build
  Isaac bundles may conflict with the pin.
- LIBERO officially predates numpy 2; it works on this stack in the tests, but
  it is outside LIBERO's supported configuration.
