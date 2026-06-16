import torch

from egcvmamba.models import build_model, build_segmentation_model


def test_classification_shape():
    model = build_model("tiny", num_classes=1000)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1000)


def test_segmentation_shape():
    model = build_segmentation_model("tiny", num_classes=150)
    model.eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 150, 128, 128)
