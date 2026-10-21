"""模型结构：简单 CNN 与复杂 CNN 的差异必须只有"卷积层数"。

这三个断言是课程口径的一部分：如果以后有人给 SimpleCNN 加残差、给 CNN 换池化，
课上"唯一变量"的说法就不成立了，测试会先炸。
"""
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from trainer.models import CNN, MLP, OWWHead, SimpleCNN, build, count_params  # noqa: E402


def count_convs(module) -> int:
    return sum(1 for m in module.modules() if isinstance(m, nn.Conv2d))


def test_simple_cnn_has_single_conv_and_pool():
    # 只数特征提取用的卷积（池化层的注意力 Conv2d(16,1,1) 两边都有）
    net = SimpleCNN()
    assert count_convs(net.features) == 1
    assert count_convs(CNN().features) == 3
    assert sum(1 for m in net.features.modules() if isinstance(m, nn.MaxPool2d)) == 1
    assert count_params(net) < count_params(CNN())
    assert count_params(net) > 0


def test_simple_cnn_shares_pooling_and_head_with_cnn():
    """除卷积块外，池化方式与输出维度必须一致，否则对比不成立。"""
    simple, full = SimpleCNN(), CNN()
    assert type(simple.pool) is type(full.pool)
    assert simple.head.out_features == full.head.out_features == 2


def test_simple_cnn_forward_shape():
    net = SimpleCNN().eval()
    with torch.no_grad():
        out = net(torch.zeros(2, 1, 40, 100))
    assert out.shape == (2, 2)


def test_build_dispatch():
    assert isinstance(build("simple"), SimpleCNN)
    assert isinstance(build("cnn"), CNN)
    assert isinstance(build("mlp"), MLP)
    assert isinstance(build("head"), OWWHead)
    with pytest.raises(SystemExit):
        build("nope")
