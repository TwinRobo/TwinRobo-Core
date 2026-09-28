# Roadmap

What TwinRobo does not do yet, and how it is meant to. Plans, not promises:
if one of these matters to you, open an issue or a pull request.

## Depth cameras

**Today:** TwinRobo simulates what cameras *image* (lens blur, distortion,
vignetting, sensor and ISP), not what depth cameras *measure*. No catalog camera
outputs depth (`outputs.depth: false`); `frame.depth` raises
`DepthUnavailableError`. `force_depth=True` still returns the simulator's
ground truth for labels and evaluation (see
[depth output](docs/cameras.md#depth-output)).

**Why not ground truth:** the simulator's depth is the exact scene depth. A
depth camera's is not, and a policy trained on exact depth meets a very
different signal on the robot. Showing ground truth as, say, a RealSense's depth
would be the gap TwinRobo exists to close.

### How depth cameras measure

| Principle | Products (examples) | What it gets wrong |
|---|---|---|
| Active IR stereo: two IR imagers + a dot projector, matched on the device | Intel RealSense D400 (D435, D455) | error grows with distance², holes where matching fails, minimum range, flying pixels at edges, invalid band where only one imager sees |
| Passive stereo: two RGB/mono eyes, matched on the host (often by a network) | Stereolabs ZED, Luxonis OAK-D | as above, and no depth on texture-less surfaces; results depend on the matcher |
| Time of flight | Azure Kinect / Orbbec Femto, ToF modules | multipath near corners, flying pixels, dark and specular dropouts, range ambiguity |

The RealSense D455, for example: 95 mm baseline, on-chip semi-global matching
(census cost), depth error around 2% at 4 m in good conditions and much worse on
low-texture surfaces or in sunlight (which washes out the projector), and a
minimum distance of about 0.5 m. The SDK can re-project depth into the RGB
camera's view ("aligned depth"), which adds occlusion gaps of its own.

### Plan

1. **Stereo depth from the module's own eyes.** A stereo module
   (`twinrobo.stereo`, e.g. `intel/realsense-d455`) already places its two eyes
   (`realsense-d455/depth`) at the right baseline, and each eye renders through
   its lens twin. Add a matcher that produces depth from the pair:
   semi-global matching with a census cost for RealSense-like modules, with its
   disparity range, subpixel precision, left-right check and filters, so holes,
   error growing with distance, edge artefacts and the minimum range follow from
   the geometry instead of being painted on.
2. **Active illumination.** Render the projector's dot pattern into the IR eyes
   (a spot light with a pattern texture in MuJoCo and Isaac Sim), with its range
   and falloff and the ambient IR that washes it out. Passive stereo modules skip
   this step.
3. **The depth pipeline's output.** Depth units and quantization, the valid
   range, the device's post-processing (spatial and temporal filters, hole
   filling), and alignment to the RGB camera with its occlusion gaps.
4. **Validation.** Captures of a real module against known targets (a plane at
   several distances, edges, low-texture and specular samples): error against
   distance, fill rate and edge profiles, as the image path is validated today.
5. **Then** set `outputs.depth: true` for those modules, meaning *this is the
   depth that camera produces*, and offer the depth view in the Studio and the
   previews again.

A cheaper, earlier step is a **parametric noise model** of a camera's depth
(error growing with distance², dropouts by surface and range, edge noise),
fitted to the vendor's specs. It is less faithful than simulating the
measurement, but useful for training robustness while the stereo path is built.
Time-of-flight cameras need their own model (multipath, phase wrapping) and come
after stereo.
