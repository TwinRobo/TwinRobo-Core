# MuJoCo

Render a TwinRobo camera in any MuJoCo scene. Start with `MujocoCameraTwin`.

## The camera

<span id="twinrobo.plugins.mujoco.camera"></span>

::: twinrobo.plugins.mujoco.camera
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [MujocoCameraTwin]

## Render backends

::: twinrobo.plugins.mujoco.camera.MujocoRenderer

::: twinrobo.plugins.mujoco.camera.RobosuiteRenderer

::: twinrobo.plugins.mujoco.camera.RenderBackend

::: twinrobo.plugins.mujoco.camera.to_uint8

## Mounting a camera

<span id="twinrobo.plugins.mujoco.mounts"></span>

::: twinrobo.plugins.mujoco.mounts
    options:
      show_root_heading: false
      show_root_toc_entry: false

## LiDAR

See [LiDAR](../cameras/lidar.md).

## Ray casting

<span id="twinrobo.plugins.mujoco.lensrender"></span>

::: twinrobo.plugins.mujoco.lensrender
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [MujocoLensScene, SceneMesh, LensRayRenderer, geom_triangles]
