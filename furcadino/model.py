"""BiFurca-DINO: a bilateral-contrast, arch-canonical detector for furcation lesions on panoramic radiographs.

Components (each can be switched off for the ablation study):
  mirror   MidlineSymmetry + ContralateralCrossAttention. A panoramic radiograph is roughly left-right symmetric
           about the facial midline. The midline column is found by maximising the left-right similarity of frozen
           foundation-model features (soft argmax, no labels). Each location then attends to a small window around its
           mirror-image location and receives both the contralateral evidence and the residual (asymmetry) feature.
  coords   Arch-canonical coordinates: signed distance to the estimated midline and normalised height, Fourier
           encoded and added to the features, so the detector can learn where in the dental arch furcations occur
           (molar region) without any tooth labels.
  context  A two-layer transformer over a coarse (stride 28) map so every location can use evidence from the other
           13 finding classes (bone loss, root treatment, restorations) detected on the same radiograph.
The backbone is DINOv2 and stays frozen, so the plain model (all three off) is exactly the DINOv2 baseline with the
same detection head and training recipe.
"""
from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScalarMix(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n))

    def forward(self, xs):
        w = torch.softmax(self.w, 0)
        return sum(wi * x for wi, x in zip(w, xs))


class FrozenDino(nn.Module):
    """DINOv2 (HF) frozen; returns a (B, C, H/14, W/14) map mixing the last `n_mix` layers."""

    def __init__(self, name="facebook/dinov2-base", n_mix=4, config=None):
        super().__init__()
        from transformers import Dinov2Model, Dinov2Config
        self.model = Dinov2Model(config) if config is not None else Dinov2Model.from_pretrained(name)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        self.n_mix = n_mix
        self.mix = ScalarMix(n_mix)
        self.out_dim = self.model.config.hidden_size
        self.patch = self.model.config.patch_size

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, x):
        B, _, H, W = x.shape
        h, w = H // self.patch, W // self.patch
        with torch.no_grad():
            out = self.model(pixel_values=x, output_hidden_states=True)
            hs = out.hidden_states[-self.n_mix:]
            hs = [self.model.layernorm(t)[:, 1:, :] for t in hs]
        hs = [t.reshape(B, h, w, -1).permute(0, 3, 1, 2).float() for t in hs]
        return self.mix(hs)


class MidlineSymmetry(nn.Module):
    """Label-free midline: soft argmax over candidate columns of left-right feature similarity.
    Returns c (B,) in feature-column units (sub-pixel) and the similarity profile."""

    def __init__(self, search=(0.30, 0.70), temp=0.02, span=0.25):
        super().__init__()
        self.search, self.temp, self.span = search, temp, span

    @torch.no_grad()
    def forward(self, f):
        g = F.normalize(f.float(), dim=1)
        B, C, H, W = g.shape
        K = max(2, int(W * self.span))
        lo = max(K, int(W * self.search[0]))
        hi = min(W - 1 - K, int(W * self.search[1]))
        cands = torch.arange(lo, hi + 1, device=g.device)
        sims = []
        for c in cands.tolist():
            left = g[..., c - K:c].flip(-1)
            right = g[..., c + 1:c + 1 + K]
            sims.append((left * right).sum(1).mean((1, 2)))
        sim = torch.stack(sims, 1)  # (B, ncand)
        wgt = torch.softmax(sim / self.temp, 1)
        c_hat = (wgt * cands.float()[None]).sum(1)
        return c_hat, sim, cands


def mirror_map(x, c):
    """Reflect feature map x (B,C,H,W) about column c (B,) (units of columns). Out of range -> 0."""
    B, C, H, W = x.shape
    xs = torch.arange(W, device=x.device, dtype=x.dtype)[None, :].expand(B, W)
    src = 2 * c[:, None].to(x.dtype) - xs  # (B, W) source column for each target column
    gx = src / (W - 1) * 2 - 1
    gy = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype)
    grid = torch.stack([gx[:, None, :].expand(B, H, W), gy[None, :, None].expand(B, H, W)], -1)
    valid = ((src >= 0) & (src <= W - 1)).to(x.dtype)[:, None, None, :]
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True) * valid, valid


class ContralateralCrossAttention(nn.Module):
    """Each location attends to a (k x k) window around its mirror-image location."""

    def __init__(self, dim, d_att=128, heads=4, k=7):
        super().__init__()
        self.k, self.h, self.d = k, heads, d_att
        self.q = nn.Conv2d(dim, d_att, 1)
        self.kv = nn.Conv2d(dim, 2 * d_att, 1)
        self.out = nn.Conv2d(2 * d_att, dim, 1)
        self.rel = nn.Parameter(torch.zeros(heads, k * k))

    def forward(self, f, m, valid):
        B, C, H, W = f.shape
        q = self.q(f)  # (B,d,H,W)
        k, v = self.kv(m).chunk(2, 1)
        pad = self.k // 2
        ku = F.unfold(k, self.k, padding=pad).view(B, self.h, self.d // self.h, self.k * self.k, H * W)
        vu = F.unfold(v, self.k, padding=pad).view(B, self.h, self.d // self.h, self.k * self.k, H * W)
        qv = q.view(B, self.h, self.d // self.h, 1, H * W)
        att = (qv * ku).sum(2) / math.sqrt(self.d // self.h) + self.rel[None, :, :, None]  # (B,h,kk,HW)
        att = torch.softmax(att, 2)
        ctx = (att[:, :, None] * vu).sum(3).reshape(B, self.d, H, W)  # (B,d,H,W)
        diff = q - ctx
        return self.out(torch.cat([ctx, diff], 1)) * valid


class ArchCoords(nn.Module):
    def __init__(self, dim, bands=6):
        super().__init__()
        self.bands = bands
        self.mlp = nn.Sequential(nn.Conv2d(3 * 2 * bands + 3, dim, 1), nn.GELU(), nn.Conv2d(dim, dim, 1))

    def coords(self, c, H, W, device):
        xs = torch.arange(W, device=device, dtype=torch.float32)[None, :]
        s = (xs - c[:, None].float()) / (W / 2)  # signed distance to midline in [-1,1]
        ys = torch.linspace(0, 1, H, device=device)
        s_map = s[:, None, :].expand(-1, H, -1)
        y_map = ys[None, :, None].expand(s.shape[0], -1, W)
        return torch.stack([s_map, s_map.abs(), y_map], 1)  # (B,3,H,W)

    def forward(self, c, H, W, device, dtype):
        co = self.coords(c, H, W, device)
        freqs = (2.0 ** torch.arange(self.bands, device=device)) * math.pi
        enc = [co]
        for fr in freqs:
            enc += [torch.sin(co * fr), torch.cos(co * fr)]
        return self.mlp(torch.cat(enc, 1).to(dtype))


class CoarseContext(nn.Module):
    def __init__(self, dim, layers=2, heads=8):
        super().__init__()
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, 0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)

    def forward(self, f):
        B, C, H, W = f.shape
        g = F.avg_pool2d(f, 2, ceil_mode=True)
        h, w = g.shape[-2:]
        t = self.enc(g.flatten(2).transpose(1, 2))
        t = t.transpose(1, 2).reshape(B, C, h, w)
        return f + F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)


class BiFurcaBackbone(nn.Module):
    """Returns an OrderedDict of three levels (strides 7, 14, 28) with `out_channels` channels for torchvision FCOS."""

    def __init__(self, dino=None, dino_name="facebook/dinov2-base", dino_config=None, dim=256,
                 mirror=True, coords=True, context=True):
        super().__init__()
        self.dino = dino or FrozenDino(dino_name, config=dino_config)
        self.out_channels = dim
        self.flags = dict(mirror=mirror, coords=coords, context=context)
        self.proj = nn.Sequential(nn.Conv2d(self.dino.out_dim, dim, 1), nn.GroupNorm(32, dim), nn.GELU())
        self.midline = MidlineSymmetry()
        self.mirror = ContralateralCrossAttention(dim) if mirror else None
        self.fuse = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.GroupNorm(32, dim), nn.GELU()) if mirror else None
        self.arch = ArchCoords(dim) if coords else None
        self.context = CoarseContext(dim) if context else None
        self.up = nn.Sequential(nn.ConvTranspose2d(dim, dim, 2, 2), nn.GroupNorm(32, dim), nn.GELU(),
                                nn.Conv2d(dim, dim, 3, padding=1))
        self.down = nn.Sequential(nn.Conv2d(dim, dim, 3, 2, 1), nn.GroupNorm(32, dim), nn.GELU())
        self.last_midline = None

    def forward(self, x):
        # bf16 only inside the backbone: box coordinates (up to ~1300) are not representable to the pixel in bf16,
        # which makes the FCOS centre-ness target negative under a square root (NaN). Heads and losses stay fp32.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            out = self._forward(x)
        return OrderedDict((k, v.float()) for k, v in out.items())

    def _forward(self, x):
        raw = self.dino(x)  # (B,C,h,w)
        B, _, h, w = raw.shape
        c, _, _ = self.midline(raw)
        self.last_midline = c.detach()
        f = self.proj(raw)
        if self.mirror is not None:
            m, valid = mirror_map(f, c)
            f = f + self.fuse(self.mirror(f, m, valid))
        if self.arch is not None:
            f = f + self.arch(c, h, w, x.device, f.dtype)
        if self.context is not None:
            f = self.context(f)
        return OrderedDict([("0", self.up(f)), ("1", f), ("2", self.down(f))])


class SoftplusRegressionHead(nn.Module):
    """torchvision's FCOS regression head ends in a ReLU, which dies at initialisation on frozen foundation features
    (all boxes collapse to zero size, observed in the fold-0 pilot). Softplus keeps a non-zero gradient everywhere."""

    def __init__(self, head):
        super().__init__()
        self.conv, self.bbox_reg, self.bbox_ctrness = head.conv, head.bbox_reg, head.bbox_ctrness
        nn.init.constant_(self.bbox_reg.bias, 2.0)  # initial box half-size ~ 2 anchor sizes

    def forward(self, x):
        reg, ctr = [], []
        for features in x:
            t = self.conv(features)
            r = F.softplus(self.bbox_reg(t))
            c = self.bbox_ctrness(t)
            N, _, H, W = r.shape
            reg.append(r.view(N, -1, 4, H, W).permute(0, 3, 4, 1, 2).reshape(N, -1, 4))
            ctr.append(c.view(N, -1, 1, H, W).permute(0, 3, 4, 1, 2).reshape(N, -1, 1))
        return torch.cat(reg, 1), torch.cat(ctr, 1)


def build_detector(variant="full", dino_name="facebook/dinov2-base", dino_config=None, num_classes=15,
                   size=(728, 1316)):
    """variant: full | plain | no_mirror | no_coords | no_context. Labels are 1..14 (0 unused)."""
    from torchvision.models.detection import FCOS
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.transform import GeneralizedRCNNTransform
    flags = dict(mirror=True, coords=True, context=True)
    if variant == "plain":
        flags = dict(mirror=False, coords=False, context=False)
    elif variant.startswith("no_"):
        flags[variant[3:]] = False
    bb = BiFurcaBackbone(dino_name=dino_name, dino_config=dino_config, **flags)
    anchors = AnchorGenerator(sizes=((7,), (14,), (28,)), aspect_ratios=((1.0,), (1.0,), (1.0,)))
    det = FCOS(bb, num_classes=num_classes, anchor_generator=anchors, score_thresh=0.01, nms_thresh=0.5,
               detections_per_img=100, topk_candidates=1000)
    det.head.regression_head = SoftplusRegressionHead(det.head.regression_head)
    det.transform = GeneralizedRCNNTransform(size[0], size[1], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
                                             size_divisible=28, fixed_size=size)
    return det
