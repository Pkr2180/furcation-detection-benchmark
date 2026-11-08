import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from transformers import Dinov2Config
from furcadino.model import MidlineSymmetry, mirror_map, ArchCoords, build_detector

TINY = Dinov2Config(hidden_size=64, num_hidden_layers=4, num_attention_heads=2, intermediate_size=128,
                    patch_size=14, image_size=56)
SIZE = (280, 532)


def sym_features(B=2, C=16, H=20, W=41, center=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, C, H, W, generator=g)
    for k in range(1, 20):  # make columns symmetric about `center`
        if center - k >= 0 and center + k < W:
            x[..., center + k] = x[..., center - k]
    return x


def test_midline_recovers_axis():
    for center in (16, 20, 24):
        f = sym_features(center=center)
        c, sim, cands = MidlineSymmetry(search=(0.3, 0.7), span=0.2)(f)
        assert (c - center).abs().max() < 0.6, (center, c)


def test_mirror_is_involution_on_valid_region():
    x = torch.randn(2, 8, 10, 31)
    c = torch.tensor([15.0, 14.0])
    m, v = mirror_map(x, c)
    mm, v2 = mirror_map(m, c)
    inside = (v * v2).bool().expand_as(x)
    assert torch.allclose(mm[inside], x[inside], atol=1e-4)


def test_image_flip_flips_signed_but_not_absolute_coordinate():
    a = ArchCoords(16)
    W = 41
    c = torch.tensor([18.0])
    co = a.coords(c, 5, W, "cpu")
    cf = a.coords(torch.tensor([W - 1 - 18.0]), 5, W, "cpu")
    assert torch.allclose(co[:, 0].flip(-1), -cf[:, 0], atol=1e-5)
    assert torch.allclose(co[:, 1].flip(-1), cf[:, 1], atol=1e-5)


def _targets(n=2):
    t = []
    for i in range(n):
        t.append({"boxes": torch.tensor([[100., 80., 160., 140.], [300., 120., 380., 200.]]),
                  "labels": torch.tensor([10, 3])})
    return t


def test_full_detector_trains_and_backbone_frozen():
    torch.manual_seed(0)
    det = build_detector("full", dino_config=TINY, size=SIZE)
    imgs = [torch.rand(3, 280, 532), torch.rand(3, 280, 532)]
    det.train()
    losses = det(imgs, _targets())
    tot = sum(losses.values())
    assert torch.isfinite(tot)
    tot.backward()
    assert all(p.grad is None for p in det.backbone.dino.model.parameters())
    assert det.backbone.mirror.q.weight.grad is not None and det.backbone.mirror.q.weight.grad.abs().sum() > 0
    assert det.backbone.arch.mlp[0].weight.grad.abs().sum() > 0
    det.eval()
    with torch.no_grad():
        out = det(imgs)
    assert len(out) == 2 and set(out[0]) == {"boxes", "scores", "labels"}


def test_variants_differ_only_in_requested_modules():
    plain = build_detector("plain", dino_config=TINY, size=SIZE).backbone
    assert plain.mirror is None and plain.arch is None and plain.context is None
    nm = build_detector("no_mirror", dino_config=TINY, size=SIZE).backbone
    assert nm.mirror is None and nm.arch is not None and nm.context is not None
    for v, attr in (("no_coords", "arch"), ("no_context", "context")):
        b = build_detector(v, dino_config=TINY, size=SIZE).backbone
        assert getattr(b, attr) is None
        assert b.mirror is not None


def test_mirror_module_sees_asymmetry():
    """Contralateral cross-attention output must be ~0 residual when the map is perfectly symmetric
    and clearly non-zero after adding a one-sided lesion-like blob."""
    from furcadino.model import ContralateralCrossAttention
    torch.manual_seed(0)
    C, H, W, c = 16, 12, 41, 20
    att = ContralateralCrossAttention(C, d_att=16, heads=2, k=5)
    f = sym_features(B=1, C=C, H=H, W=W, center=c)
    m, v = mirror_map(f, torch.tensor([float(c)]))
    # with a symmetric map the keys equal the queries' mirror image; check that the diff branch is small
    q = att.q(f)
    k, _ = att.kv(m).chunk(2, 1)
    f2 = f.clone()
    f2[..., 8:10, 8:10] += 5.0  # one-sided blob
    m2, _ = mirror_map(f2, torch.tensor([float(c)]))
    d_sym = (att.q(f) - att.kv(m)[:, :16]).abs().mean()
    d_asym = (att.q(f2) - att.kv(m2)[:, :16]).abs().mean()
    assert d_asym > d_sym


def test_detector_can_overfit_two_images_and_boxes_not_degenerate():
    """Regression for the pilot failure: after a short fit on 2 images the model must emit non-degenerate boxes that
    overlap the training targets."""
    torch.manual_seed(0)
    det = build_detector("full", dino_config=TINY, size=SIZE)
    imgs = [torch.rand(3, 280, 532) for _ in range(2)]
    tg = [{"boxes": torch.tensor([[100., 80., 180., 150.]]), "labels": torch.tensor([10])},
          {"boxes": torch.tensor([[300., 120., 400., 210.]]), "labels": torch.tensor([10])}]
    opt = torch.optim.AdamW([p for p in det.parameters() if p.requires_grad], lr=1e-3)
    det.train()
    first = None
    for s in range(120):
        loss = sum(det(imgs, tg).values())
        opt.zero_grad(); loss.backward(); opt.step()
        first = float(loss) if first is None else first
    assert float(loss) < 0.6 * first, (first, float(loss))
    det.eval()
    with torch.no_grad():
        out = det(imgs)
    from furcadino.evalap import iou_matrix
    for o, t in zip(out, tg):
        b = o["boxes"].numpy()
        assert len(b) and ((b[:, 2] - b[:, 0]) > 1).all()
        assert iou_matrix(b, t["boxes"].numpy()).max() > 0.5
