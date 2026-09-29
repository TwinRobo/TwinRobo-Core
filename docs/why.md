# Why TwinRobo?

Robot policies are trained on simulator images, and simulators render through an
ideal pinhole camera. The robot sees through a real one: another field of view,
lens distortion, blur, vignetting, a monochrome or rolling-shutter sensor. A
policy that works in simulation meets a different image on the robot.

We measured how much that matters for a vision-language-action model.

## The same policy, the same scene, another camera

OpenVLA-7B fine-tuned on LIBERO-Spatial runs the standard LIBERO evaluation
(OpenVLA's own evaluation code: fixed initial states, a 220-step limit). The
only change is the camera it sees through: LIBERO's pinhole, which it was
trained on, or a TwinRobo twin of a real camera at the same pose, with that
camera's field of view and lens.

| Camera | Field of view | Success | 95% interval | vs. pinhole | Episodes |
|---|---|---|---|---|---|
| Pinhole (LIBERO, the training camera) | 45° vertical | **85%** | 77–91% | – | 100 |
| Logitech C920 | 70° × 43° | **84%** | 76–90% | -1 pts | 100 |
| Luxonis OAK-D mono | 72° × 49°, monochrome | **65%** | 55–74% | -20 pts | 100 |
| Intel RealSense D455 RGB | 90° × 64° | **0%** | 0–4% | -85 pts | 100 |
| Stereolabs ZED X 2.2 mm | 110° × 80° | **0%** | 0–4% | -85 pts | 100 |

Cameras framed close to the training camera hold up; the wide ones break the
policy entirely. The scene is framed twice as wide, the objects cover a few
pixels, and the arm never reaches them. The monochrome OAK-D, with a field of
view close to the C920's, still loses 20 points.

## What it looks like

Each panel is the 256 × 256 image OpenVLA saw, from the same starting state. A
green frame marks a completed task, a red one the time limit.

<figure class="tr-why-video" markdown="0">
  <video src="../assets/why/openvla-spatial-task3.mp4" autoplay muted loop playsinline preload="metadata" aria-label="Task 3 through five cameras"></video>
  <figcaption>Task 3: pick up the black bowl on the cookie box and place it on the plate.</figcaption>
</figure>

<figure class="tr-why-video" markdown="0">
  <video src="../assets/why/openvla-spatial-task4.mp4" autoplay muted loop playsinline preload="metadata" aria-label="Task 4 through five cameras"></video>
  <figcaption>Task 4: pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate.</figcaption>
</figure>

<figure class="tr-why-video" markdown="0">
  <video src="../assets/why/openvla-spatial-task5.mp4" autoplay muted loop playsinline preload="metadata" aria-label="Task 5 through five cameras"></video>
  <figcaption>Task 5: pick up the black bowl on the ramekin and place it on the plate.</figcaption>
</figure>

## What TwinRobo does about it

- **Evaluate with the camera you will deploy.** Put a catalog camera, or your
  own, on the robot in simulation and see what your policy does with it,
  before buying hardware. [Try it in a simulator](playground.md).
- **Train with it.** Render demonstrations and training data through the real
  camera's optics, in MuJoCo, LIBERO, RoboCasa or Isaac Sim, so the policy
  learns the images it will get. [Get started](getting-started.md).
- **Choose cameras on evidence.** Compare cameras' fields of view, lenses and
  sensors on your own task, side by side. [Browse the catalog](catalog.md).

## About these results

- Ten initial states per task and camera (the official protocol uses 50), all
  ten LIBERO-Spatial tasks; the intervals are 95% Wilson intervals.
- Each camera is rendered with TwinRobo's PSF method from LIBERO's agent-view
  pose, then center-cropped and resized to the policy's 256 × 256 input.
- These are simulation results: they measure the gap between the pinhole a
  policy is trained on and a real camera's optics, not every source of
  sim-to-real difference.
- A second pass, which keeps LIBERO's field of view and applies only each real
  camera's lens, will separate framing from optics.
