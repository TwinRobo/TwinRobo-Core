---
hide:
  - navigation
  - toc
---

# Playground

Every camera in the [catalog](catalog.md), seen from a robot's wrist in two
simulators: the same pose and scene, rendered through each camera's own lens,
field of view and focus. Drag the slider to compare a camera's image with the
simulator's ideal pinhole view at the same field of view.

<div id="tr-playground" data-base="../assets/playground/" markdown="0">
  <div class="trp-bar">
    <div class="trp-group" role="tablist" aria-label="Scene" data-key="scene"></div>
    <div class="trp-group" aria-label="Rendering method" data-key="method"></div>
    <div class="trp-group" aria-label="View" data-key="view"></div>
  </div>
  <div class="trp-main">
    <div class="trp-stage">
      <div class="trp-frame">
        <img class="trp-under" alt="">
        <div class="trp-over"><img alt=""></div>
        <canvas class="trp-diff" hidden></canvas>
        <input class="trp-slider" type="range" min="0" max="100" value="50" aria-label="Compare">
        <span class="trp-tag trp-tag-l"></span><span class="trp-tag trp-tag-r"></span>
      </div>
      <p class="trp-note"></p>
    </div>
    <aside class="trp-info"></aside>
  </div>
  <h2 class="trp-h">All cameras</h2>
  <div class="trp-grid"></div>
  <noscript>The playground needs JavaScript.</noscript>
</div>

## What you are looking at

- **Pinhole** is what the simulator renders by itself: a perfect lens with the
  camera's field of view.
- **PSF bank** blurs that image with the lens' point-spread functions,
  measured by tracing the lens across the field and at each depth, and then
  applies its distortion and vignetting. It is the fastest method and the default.
- **Pupil views** render the scene from several points on the lens' entrance
  pupil and combine them along each pixel's traced rays, so blur follows the
  scene's occlusions.
- **Ray cast** traces every pixel's rays through the lens and casts them into
  the scene's geometry: the most exact method.
- **Difference** shows where a camera's image departs from the pinhole's;
  **Depth** is the depth map a depth camera outputs (near is warm): it is
  offered only for cameras that have one, the RealSense D400 series.

Each image is 640 px wide, read out at the camera's own aspect ratio. The
cameras are rendered offline with
[`tools/docs/playground`](https://github.com/TwinRobo/TwinRobo-Core/tree/main/tools/docs/playground):
MuJoCo through robosuite's PickPlace task, Isaac Sim 6.1 through a random
tabletop. To render your own scene, see [Get started](getting-started.md).
