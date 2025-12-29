"""
MOUAADNET-ULTRA: Complete Test Suite
=====================================
Validates all components of the neural network architecture.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import time


def test_backbone():
    """Test backbone components."""
    print("\n" + "=" * 60)
    print("🔵 Phase 1: Testing Backbone Components")
    print("=" * 60)
    
    from mouaadnet_ultra.backbone import PConv, GhostModule, InvertedResidualBlock, NanoBackbone
    
    x = torch.randn(2, 64, 32, 32)
    x_rgb = torch.randn(2, 3, 416, 416)
    
    # PConv
    pconv = PConv(64)
    out = pconv(x)
    print(f"✓ PConv: {x.shape} -> {out.shape}")
    
    # Ghost Module
    ghost = GhostModule(64, 128)
    out = ghost(x)
    print(f"✓ GhostModule: {x.shape} -> {out.shape}")
    
    # IRB
    irb = InvertedResidualBlock(64, 64)
    out = irb(x)
    print(f"✓ InvertedResidualBlock: {x.shape} -> {out.shape}")
    
    # NanoBackbone
    backbone = NanoBackbone()
    p3, p4, p5 = backbone(x_rgb)
    print(f"✓ NanoBackbone: {x_rgb.shape} -> P3={p3.shape}, P4={p4.shape}, P5={p5.shape}")
    
    return True


def test_neck():
    """Test neck components."""
    print("\n" + "=" * 60)
    print("🟡 Phase 2: Testing Neck Components")
    print("=" * 60)
    
    from mouaadnet_ultra.neck import SlimPAN, SPPLite, CSPBlock
    
    # Simulate backbone outputs
    p3 = torch.randn(2, 24, 52, 52)
    p4 = torch.randn(2, 40, 26, 26)
    p5 = torch.randn(2, 128, 13, 13)
    
    # SPP-Lite
    spp = SPPLite(128, 128)
    out = spp(p5)
    print(f"✓ SPPLite: {p5.shape} -> {out.shape}")
    
    # CSP Block
    csp = CSPBlock(64, 64)
    out = csp(torch.randn(2, 64, 32, 32))
    print(f"✓ CSPBlock: (2, 64, 32, 32) -> {out.shape}")
    
    # Slim-PAN
    pan = SlimPAN((24, 40, 128), 64)
    n3, n4, n5 = pan((p3, p4, p5))
    print(f"✓ SlimPAN: -> N3={n3.shape}, N4={n4.shape}, N5={n5.shape}")
    
    return True


def test_heads():
    """Test head components."""
    print("\n" + "=" * 60)
    print("🔴 Phase 3: Testing Head Components")
    print("=" * 60)
    
    from mouaadnet_ultra.heads import DetectionHead, GenderHead, DecoupledHead
    
    features = (
        torch.randn(2, 64, 52, 52),
        torch.randn(2, 64, 26, 26),
        torch.randn(2, 64, 13, 13),
    )
    
    # Detection Head
    det_head = DetectionHead(64)
    outputs = det_head(features)
    print(f"✓ DetectionHead: 3 scales -> {len(outputs['heatmaps'])} heatmaps")
    
    # Gender Head
    gender_head = GenderHead(64)
    out = gender_head(torch.randn(4, 64, 16, 16))
    print(f"✓ GenderHead: (4, 64, 16, 16) -> {out.shape}")
    
    # Decoupled Head
    decoupled = DecoupledHead(64)
    outputs = decoupled(features)
    print(f"✓ DecoupledHead: -> heatmaps:{len(outputs['heatmaps'])}, gender:{outputs['gender'].shape}")
    
    return True


def test_losses():
    """Test loss functions."""
    print("\n" + "=" * 60)
    print("🟢 Phase 4a: Testing Loss Functions")
    print("=" * 60)
    
    from mouaadnet_ultra.losses import CIoULoss, FocalLoss, MultiTaskLoss
    
    # CIoU Loss
    pred_boxes = torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60]], dtype=torch.float)
    target_boxes = torch.tensor([[10, 10, 50, 50], [25, 25, 65, 65]], dtype=torch.float)
    
    ciou = CIoULoss()
    loss = ciou(pred_boxes, target_boxes)
    print(f"✓ CIoULoss: {loss.item():.4f}")
    
    # Focal Loss
    focal = FocalLoss()
    pred = torch.tensor([0.9, 0.1])
    target = torch.tensor([1.0, 0.0])
    loss = focal(pred, target)
    print(f"✓ FocalLoss: {loss.item():.4f}")
    
    # Multi-Task Loss
    multi_loss = MultiTaskLoss()
    predictions = {
        'heatmaps': [torch.sigmoid(torch.randn(2, 1, 52, 52))],
        'sizes': [torch.randn(2, 2, 52, 52)],
        'offsets': [torch.randn(2, 2, 52, 52)],
        'gender': torch.randn(2, 1),
    }
    targets = {
        'heatmaps': [torch.zeros(2, 1, 52, 52)],
        'sizes': [torch.zeros(2, 2, 52, 52)],
        'offsets': [torch.zeros(2, 2, 52, 52)],
        'gender_labels': torch.randint(0, 2, (2, 1)).float(),
    }
    losses = multi_loss(predictions, targets)
    print(f"✓ MultiTaskLoss: total={losses['total'].item():.4f}")
    
    return True


def test_optimization():
    """Test optimization components."""
    print("\n" + "=" * 60)
    print("🟢 Phase 4b: Testing Optimization Components")
    print("=" * 60)
    
    from mouaadnet_ultra.optim import RepVGGBlock, OneCycleLR
    import copy
    
    # RepVGG Block
    x = torch.randn(2, 64, 32, 32)
    rep_block = RepVGGBlock(64, 64)
    rep_block.eval()
    
    out_train = rep_block(x)
    
    # Fuse and compare
    rep_fused = copy.deepcopy(rep_block)
    rep_fused.fuse()
    out_fused = rep_fused(x)
    
    diff = (out_train - out_fused).abs().max().item()
    print(f"✓ RepVGGBlock fusion: max diff = {diff:.6f}")
    assert diff < 1e-4, "Fusion error!"
    
    # One-Cycle LR
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=0.01, total_steps=100)
    
    lrs = []
    for _ in range(100):
        lrs.append(scheduler.get_lr()[0])
        scheduler.step()
    
    print(f"✓ OneCycleLR: init={lrs[0]:.6f}, max={max(lrs):.6f}, final={lrs[-1]:.6f}")
    
    return True


def test_complete_model():
    """Test complete model."""
    print("\n" + "=" * 60)
    print("⭐ Testing Complete Model")
    print("=" * 60)
    
    from mouaadnet_ultra.model import MouaadNetUltra, MouaadNetUltraLite
    
    # Standard model
    model = MouaadNetUltra()
    model.eval()
    
    x = torch.randn(1, 3, 416, 416)
    
    with torch.no_grad():
        outputs = model(x)
    
    print(f"✓ MouaadNetUltra forward pass successful")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  FP32 Size: {model.get_model_size_mb('fp32'):.2f} MB")
    print(f"  INT8 Size: {model.get_model_size_mb('int8'):.2f} MB")
    
    # Benchmark
    times = []
    for _ in range(10):
        start = time.time()
        with torch.no_grad():
            _ = model(x)
        times.append((time.time() - start) * 1000)
    
    print(f"  CPU Inference: {sum(times)/len(times):.1f} ms avg")
    
    # Lite model
    lite = MouaadNetUltraLite()
    print(f"✓ MouaadNetUltraLite: {lite.count_parameters():,} params, {lite.get_model_size_mb('int8'):.2f} MB")
    
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("🏗️  MOUAADNET-ULTRA: Complete Test Suite")
    print("    Lead Architect: MOUAAD IDOUFKIR")
    print("=" * 60)
    
    tests = [
        ("Backbone", test_backbone),
        ("Neck", test_neck),
        ("Heads", test_heads),
        ("Losses", test_losses),
        ("Optimization", test_optimization),
        ("Complete Model", test_complete_model),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed, None))
        except Exception as e:
            results.append((name, False, str(e)))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed, error in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if error:
            print(f"         Error: {error}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
