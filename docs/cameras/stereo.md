# Stereo

Two eyes that move together, such as a ZED X.

!!! note "This release's text below"
    The helper for expanding a mount is `expand_mount`. A pair is row-aligned only with parallel
    eyes, matching intrinsics and a horizontal baseline.

## Catalog stereo modules

Load a module with `ModuleRegistry().load(id)`.

{{ catalog_modules }}

## Modules

<span id="twinrobo.stereo"></span>

::: twinrobo.stereo
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [StereoModule, Eye, ModuleRegistry]

## Eye helpers

::: twinrobo.stereo.eye_poses

::: twinrobo.stereo.eye_name
