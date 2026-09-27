# Phase 1 optics benchmark

- GPU: NVIDIA GeForce RTX 3090; torch 2.10.0+cu128; DeepLens 2.5.4+278e4f720a4549187a51edaea57bb658548911ce; Python 3.12.14
- Lens: `cellphone80deg.json`, focus 2.0 m; image 1920x1200; PSF grid 17x11; 16 depth layers in [0.3, 20.0] m; ks 65
- PSF bank build (DeepLens, one-time, cached): 22.9 s
- Monte Carlo floor (two independently sampled banks, slanted plane): 61.9 dB

| Scene | Runtime render (ms) | Peak GPU mem (GB) | DeepLens reference (s) | PSNR vs reference (dB) | Ideal vs reference (dB) | Energy out/in | Reference out/in |
|---|---|---|---|---|---|---|---|
| slanted plane 0.4-15 m | 473 | 1.92 | 23.7 | 46.2 | 25.4 | 0.9991 | 1.0000 |
| fronto plane @ focus (2 m) | 41 | 1.82 | 23.4 | 46.4 | 24.9 | 0.9991 | 1.0000 |
| fronto plane @ 0.5 m | 73 | 2.01 | 23.2 | 45.9 | 25.9 | 0.9991 | 1.0000 |
| occluder 0.5 m over inf | 115 | 2.01 | 23.5 | 46.4 | 24.4 | 0.9991 | 1.0000 |

## Point responses vs direct DeepLens traces

A single bright pixel is rendered at an arbitrary field position and depth (between
PSF nodes and depth layers). Its green response is compared with a PSF traced
directly by DeepLens there. L1 = sum|a - b| (0 identical, 2 disjoint). The MC floor
is two independent traces of the same point. Nearest is the closest bank PSF.

| Pixel (row, col) | Field (x, y) | Depth (m) | L1 rendered | L1 MC floor | L1 nearest | Peak ratio |
|---|---|---|---|---|---|---|
| (600, 960) | (0.001, -0.001) | 10.0 | 0.062 | 0.076 | 0.077 | 1.01 |
| (600, 960) | (0.001, -0.001) | 1.2 | 0.038 | 0.064 | 0.093 | 0.98 |
| (600, 960) | (0.001, -0.001) | 0.33 | 0.079 | 0.094 | 0.112 | 0.92 |
| (307, 331) | (-0.655, 0.488) | 10.0 | 0.099 | 0.080 | 0.202 | 0.96 |
| (307, 331) | (-0.655, 0.488) | 1.2 | 0.086 | 0.072 | 0.329 | 0.98 |
| (307, 331) | (-0.655, 0.488) | 0.33 | 0.079 | 0.066 | 0.181 | 0.98 |
| (1150, 1900) | (0.98, -0.917) | 10.0 | 0.157 | 0.121 | 0.223 | 0.99 |
| (1150, 1900) | (0.98, -0.917) | 1.2 | 0.127 | 0.063 | 0.255 | 0.96 |
| (1150, 1900) | (0.98, -0.917) | 0.33 | 0.142 | 0.042 | 0.265 | 1.10 |
| (10, 15) | (-0.984, 0.983) | 10.0 | 0.129 | 0.085 | 0.147 | 0.95 |
| (10, 15) | (-0.984, 0.983) | 1.2 | 0.109 | 0.080 | 0.163 | 1.05 |
| (10, 15) | (-0.984, 0.983) | 0.33 | 0.121 | 0.036 | 0.152 | 1.09 |

## Notes

The reference is DeepLens `psf_map_rgb` + `conv_psf_map_depth_interp` at the same
depth layers. It uses one PSF per block and blends by output-pixel depth, with no
occlusion handling, so it is not ground truth at depth edges. Distortion and
vignetting are not modeled by either path.

Energy out/in < 1 for the runtime renderer is expected. DeepLens PSFs are
recentered on the chief ray, but coma moves their energy centroid radially
(about 1 px at the field edge), and the scatter formulation moves light with it, so a
little light leaves the frame. The gather-style reference normalizes per output pixel.

Render time scales with the number of occupied depth layers. A single plane
occupies one layer. The slanted plane occupies all layers.
