"""GPU dense-depth PSF renderer.

Turns an ideal RGB-D observation into the optical image on the sensor using a
precomputed `PSFBank`. It never calls DeepLens. PSFs come from the bank, which
an optics backend builds offline.

Why this lives in TwinRobo rather than DeepLens (§29.6): DeepLens owns PSF
computation. It also ships simple renderers, but none that combine what the
runtime needs:

- **Spatially varying PSFs without seams.** Scatter formulation
  ``out = sum_i PSF_i * (w_i . img)`` with separable bilinear weights ``w_i``
  that sum to 1. Every source point is spread by a smoothly interpolated
  PSF, and energy is conserved. Each node's weighted window is convolved by
  FFT and overlap-added onto the output.
- **Soft depth assignment.** Tent weights in disparity between adjacent depth
  layers. Invalid depth (``inf``/``nan``/``<= 0``) goes to the farthest layer,
  out-of-range depth is clamped to the nearest/farthest layer.
- **Occlusion-aware compositing.** Normalized layered compositing after
  Ikoma et al., "Depth from Defocus with Learned Optics for Imaging and
  Occlusion-aware Depth Estimation", ICCP 2021. For layer ``k`` (far -> near)
  with weights ``a_k`` and cumulative ``c_k = sum_{k' <= k} a_k'``::

      I_k = H_k(rgb * a_k) * H_k(1) / H_k(c_k)       A_k = H_k(a_k) / H_k(c_k)
      out = sum_k I_k * prod_{k' > k} (1 - A_k')

  ``H_k`` is the spatially varying blur of layer ``k``. Normalizing by the
  coverage ``H_k(c_k) / H_k(1)`` fills in background hidden behind occluders,
  which avoids dark halos, and it preserves energy for depths between layers.
  Without occlusion (``c_k = 1``) the factor is exactly 1, so the output is the
  plain scatter blur ``H_k(rgb * a_k)``. With spatially varying PSFs
  ``H_k(1) != 1``, so the original ``1 / H_k(c_k)`` would rescale the image.
  ``H_k(1)`` does not depend on the image and is cached per layer.

Not modeled here (the PSFs are recentered on the chief ray and normalized):
geometric distortion and vignetting. Those are separate stages. Energy moves
with each PSF's centroid, however. Coma shifts the centroids of off-axis PSFs
radially relative to the chief ray (about 1 px at the field edge for the
example lens), so a uniform scene is not exactly uniform (under 1%), and a
little light leaves the frame (about 0.1%). A gather-style renderer such as
DeepLens' PSF-map convolution normalizes per output pixel and hides this.

Conventions follow `twinrobo.frame` and `twinrobo.optics.psf`: PSF center
at index ``ks // 2``, with image borders reflect-padded as in DeepLens.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from ..frame import check_depth, check_rgb, valid_depth_mask
from .base import OpticsModel
from .psf import PSFBank


def _fft_size(n: int) -> int:
    """Smallest 2,3,5-smooth integer >= n (fast cuFFT sizes)."""
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def _axis_weights(size: int, n_nodes: int, pad_lo: int, pad_hi: int, device) -> Tensor:
    """Bilinear node weights along one axis of the padded image, shape ``[n_nodes, padded]``.

    Node ``i`` sits at pixel-edge coordinate ``i * size / (n - 1)`` (see
    `twinrobo.optics.psf`), i.e. pixel-index coordinate ``i * size / (n - 1) - 0.5``.
    Beyond the outermost nodes (padding only) the border node has weight 1, so
    the weights sum to 1 everywhere.
    """
    padded = size + pad_lo + pad_hi
    q = torch.arange(padded, device=device, dtype=torch.float64) - pad_lo
    if n_nodes == 1:
        return torch.ones(1, padded, device=device, dtype=torch.float32)
    step = size / (n_nodes - 1)
    u = ((q + 0.5) / step).clamp(0, n_nodes - 1)
    i0 = u.floor().clamp(max=n_nodes - 2).long()
    f = u - i0
    w = torch.zeros(n_nodes, padded, device=device, dtype=torch.float64)
    cols = torch.arange(padded, device=device)
    w[i0, cols] = 1 - f
    w[i0 + 1, cols] += f
    return w.float()


def _windows(w: Tensor) -> tuple[Tensor, int]:
    """Common window length ``R`` and per-node start covering each node's weight support."""
    padded = w.shape[1]
    nz = w > 0
    idx = torch.arange(padded, device=w.device)
    first = torch.where(nz, idx, padded).amin(dim=1)
    last = torch.where(nz, idx, -1).amax(dim=1)
    R = int((last - first + 1).max())
    starts = torch.minimum(first, torch.full_like(first, padded - R))
    return starts, R


class DenseDepthRenderer(OpticsModel):
    """Occlusion-aware, spatially varying, depth-dependent PSF rendering on the GPU.

    Args:
        bank: PSF bank. Its ``sensor_resolution`` must equal the image size.
        eps: Normalization threshold. Where the blurred cumulative alpha is
            below ``eps``, a layer contributes nothing. This avoids amplifying
            FFT round-off, and it drops at most ~``eps`` of a pixel's energy.
        padding: ``F.pad`` mode for image borders (``"reflect"`` or ``"replicate"``).
    """

    def __init__(self, bank: PSFBank, eps: float = 1e-4, padding: str = "reflect"):
        self.bank = bank
        self.eps = eps
        self.padding = padding
        self._geometry: dict[tuple, Any] = {}
        self._unit: dict[tuple, Tensor] = {}  # H_k(1) per (H, W, device, k)

    # ------------------------------------------------------------------
    def _geom(self, H: int, W: int, device) -> dict[str, Any]:
        key = (H, W, str(device))
        if key not in self._geometry:
            ks = self.bank.ks
            pt, pb = (ks - 1) // 2, ks // 2  # DeepLens padding: PSF center at ks // 2
            gw, gh = self.bank.grid
            wy = _axis_weights(H, gh, pt, pb, device)
            wx = _axis_weights(W, gw, pt, pb, device)
            sy, Rh = _windows(wy)
            sx, Rw = _windows(wx)
            row_idx = sy[:, None] + torch.arange(Rh, device=device)  # [Gh, Rh]
            col_idx = sx[:, None] + torch.arange(Rw, device=device)  # [Gw, Rw]
            self._geometry[key] = {
                "pt": pt,
                "pb": pb,
                "Rh": Rh,
                "Rw": Rw,
                "sy": sy.tolist(),
                "wy_win": wy.gather(1, row_idx),  # [Gh, Rh]
                "wx_win": wx.gather(1, col_idx),  # [Gw, Rw]
                "col_idx": col_idx,
                "out_col_idx": (sx[:, None] + torch.arange(Rw + ks - 1, device=device)).reshape(-1),
                "fft": (_fft_size(Rh + ks - 1), _fft_size(Rw + ks - 1)),
            }
        return self._geometry[key]

    def _layer_weights(self, depth: Tensor) -> Tensor:
        """Soft assignment of pixels to depth layers, ``[B, K, H, W]`` summing to 1."""
        disp_k = (1.0 / self.bank.depths_m).to(depth)  # increasing: far -> near
        K = disp_k.numel()
        valid = valid_depth_mask(depth)
        disp = torch.where(valid, 1.0 / depth.clamp_min(1e-9), disp_k[0])
        disp = disp.clamp(disp_k[0], disp_k[-1])
        if K == 1:
            return torch.ones_like(depth)
        hi = torch.searchsorted(disp_k, disp.flatten().contiguous(), right=True)
        hi = hi.clamp(1, K - 1).view_as(disp)
        lo = hi - 1
        f = ((disp - disp_k[lo]) / (disp_k[hi] - disp_k[lo])).clamp(0, 1)
        alpha = torch.zeros(
            depth.shape[0], K, *depth.shape[-2:], device=depth.device, dtype=depth.dtype
        )
        alpha.scatter_add_(1, lo, 1 - f)
        alpha.scatter_add_(1, hi, f)
        return alpha

    def _blur_layer(self, x: Tensor, k: int, g: dict[str, Any]) -> Tensor:
        """Spatially varying blur of padded stack ``x [B, G, C, Hp, Wp]`` with layer ``k`` PSFs.

        Returns the overlap-added canvas ``[B, G, C, Hp + ks - 1, Wp + ks - 1]``.
        """
        B, G, C, Hp, Wp = x.shape
        ks = self.bank.ks
        Rh, Rw = g["Rh"], g["Rw"]
        fh, fw = g["fft"]
        gw, gh = self.bank.grid
        canvas = x.new_zeros(B, G, C, Hp + ks - 1, Wp + ks - 1)
        psf_k = self.bank.psfs[k]  # [Gh, Gw, C, ks, ks]
        for i in range(gh):
            s = g["sy"][i]
            rows = x[..., s : s + Rh, :] * g["wy_win"][i][:, None]  # [B, G, C, Rh, Wp]
            win = rows[..., g["col_idx"]]  # [B, G, C, Rh, Gw, Rw]
            win = win * g["wx_win"]
            win = win.permute(0, 1, 2, 4, 3, 5)  # [B, G, C, Gw, Rh, Rw]
            f_img = torch.fft.rfft2(win, s=(fh, fw))
            f_psf = torch.fft.rfft2(psf_k[i].permute(1, 0, 2, 3), s=(fh, fw))  # [C, Gw, ...]
            out = torch.fft.irfft2(f_img * f_psf, s=(fh, fw))
            out = out[..., : Rh + ks - 1, : Rw + ks - 1]  # [B, G, C, Gw, Rh', Rw']
            out = out.permute(0, 1, 2, 4, 3, 5).reshape(B, G, C, Rh + ks - 1, -1)
            canvas[..., s : s + Rh + ks - 1, :].index_add_(-1, g["out_col_idx"], out)
        return canvas

    def _unit_response(self, k: int, H: int, W: int, C: int, g: dict[str, Any], crop) -> Tensor:
        """``H_k(1)`` cropped to the image, ``[1, C, H, W]`` (cached)."""
        key = (H, W, str(self.bank.device), k)
        if key not in self._unit:
            pt, pb = g["pt"], g["pb"]
            ones = torch.ones(1, 1, C, H + pt + pb, W + pt + pb, device=self.bank.device)
            with torch.no_grad():
                self._unit[key] = self._blur_layer(ones, k, g)[:, 0, :, crop[0], crop[1]]
        return self._unit[key]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def render(self, rgb: Tensor, depth: Tensor, metadata: dict[str, Any] | None = None) -> Tensor:
        return self.render_differentiable(rgb, depth)

    def render_differentiable(self, rgb: Tensor, depth: Tensor) -> Tensor:
        """`render` without ``no_grad``: gradients flow to ``rgb`` (and to ``bank.psfs``
        through the blurred layers). The cached unit response ``H_k(1)`` is detached, so
        rebuild the renderer if the bank PSFs change."""
        check_rgb(rgb)
        check_depth(depth, rgb)
        B, C, H, W = rgb.shape
        if (W, H) != self.bank.sensor_resolution:
            raise ValueError(
                f"Image {W}x{H} does not match PSF bank sensor resolution "
                f"{self.bank.sensor_resolution[0]}x{self.bank.sensor_resolution[1]}"
            )
        if C != self.bank.channels:
            raise ValueError(f"rgb has {C} channels, PSF bank has {self.bank.channels}")

        dtype = torch.float32
        rgb = rgb.to(self.bank.device, dtype)
        depth = depth.to(self.bank.device, dtype)
        g = self._geom(H, W, rgb.device)
        pt, pb = g["pt"], g["pb"]
        ks = self.bank.ks
        c = ks // 2
        crop = (slice(pt + c, pt + c + H), slice(pt + c, pt + c + W))

        alpha = self._layer_weights(depth)  # [B, K, H, W]
        cum = alpha.cumsum(dim=1)
        occupied = (alpha.amax(dim=(0, 2, 3)) > 0).tolist()  # single host sync

        out = torch.zeros_like(rgb)
        trans = torch.ones_like(rgb)
        for k in reversed(range(self.bank.num_depths)):  # near -> far
            if not occupied[k]:
                continue
            a = alpha[:, k : k + 1]
            stack = torch.stack(
                [rgb * a, a.expand(B, C, H, W), cum[:, k : k + 1].expand(B, C, H, W)], 1
            )
            stack = F.pad(stack.flatten(1, 2), (pt, pb, pt, pb), mode=self.padding)
            blurred = self._blur_layer(stack.view(B, 3, C, *stack.shape[-2:]), k, g)
            blurred = blurred[..., crop[0], crop[1]]
            I_k, A_k, E_k = blurred.unbind(1)
            U_k = self._unit_response(k, H, W, C, g, crop)
            ok = E_k > self.eps
            E_safe = torch.where(ok, E_k, torch.ones_like(E_k))
            I_n = torch.where(ok, I_k * U_k / E_safe, torch.zeros_like(I_k))
            A_n = torch.where(ok, (A_k / E_safe).clamp(0, 1), torch.zeros_like(A_k))
            out = out + trans * I_n
            trans = trans * (1 - A_n)
        return out

    def describe(self) -> str:
        gw, gh = self.bank.grid
        return f"DenseDepthRenderer(K={self.bank.num_depths}, grid={gw}x{gh}, ks={self.bank.ks})"
