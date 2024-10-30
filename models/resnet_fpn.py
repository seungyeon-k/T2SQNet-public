from models.resnet import ResNet, BasicBlock, Bottleneck
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights

def get_resnet_fpn_backbone(
    backbone_name,
    use_bn=False,
    pretrained=True,
    trainable_layers=3,
    returned_layers=[1, 2, 3, 4],
):
    assert backbone_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101']
    assert trainable_layers in [0, 1, 2, 3, 4, 5]
    assert returned_layers <= [1, 2, 3, 4]
    
    if backbone_name == "resnet18":
        resnet = ResNet(BasicBlock, [2, 2, 2, 2], use_bn=use_bn, input_dim=3, avg_pool_size=1)
    elif backbone_name == "resnet34":
        resnet = ResNet(BasicBlock, [3, 4, 6, 3], use_bn=use_bn, input_dim=3, avg_pool_size=1)
    elif backbone_name == "resnet50":
        resnet = ResNet(Bottleneck, [3, 4, 6, 3], use_bn=use_bn, input_dim=3, avg_pool_size=1)
    elif backbone_name == "resnet101":
        resnet = ResNet(Bottleneck, [3, 4, 23, 3], use_bn=use_bn, input_dim=3, avg_pool_size=1)
    
    # if backbone_name == "resnet18":
    #     resnet = ResNet(BasicBlock, [2, 2, 2, 2], use_bn=use_bn)
    # elif backbone_name == "resnet34":
    #     resnet = ResNet(BasicBlock, [3, 4, 6, 3], use_bn=use_bn)
    # elif backbone_name == "resnet50":
    #     resnet = ResNet(Bottleneck, [3, 4, 6, 3], use_bn=use_bn)
    # elif backbone_name == "resnet101":
    #     resnet = ResNet(Bottleneck, [3, 4, 23, 3], use_bn=use_bn)
        
    if pretrained:
        if backbone_name == 'resnet18':
            weights = ResNet18_Weights.DEFAULT
        elif backbone_name == 'resnet34':
            weights = ResNet34_Weights.DEFAULT
        elif backbone_name == 'resnet50':
            weights = ResNet50_Weights.DEFAULT
        elif backbone_name == 'resnet101':
            weights = ResNet101_Weights.DEFAULT
        resnet.load_state_dict(weights.get_state_dict(progress=True), strict=False)
    resnet = resnet.float()
    
    backbone = _resnet_fpn_extractor(
    resnet,
    trainable_layers,
    returned_layers
    )
        
    return backbone

def _resnet_fpn_extractor(
    backbone,
    trainable_layers,
    returned_layers=None,
    extra_blocks=None,
    norm_layer=None,
):

    # select layers that wont be frozen
    if trainable_layers < 0 or trainable_layers > 5:
        raise ValueError(f"Trainable layers should be in the range [0,5], got {trainable_layers}")
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers]
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)
    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()
    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    if min(returned_layers) <= 0 or max(returned_layers) >= 5:
        raise ValueError(f"Each returned layer should be in the range [1,4]. Got {returned_layers}")
    return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}
    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    return BackboneWithFPN(
        backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks, norm_layer=norm_layer
    )