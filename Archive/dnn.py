from dnn import FaceAnalyzer
from omegaconf import OmegaConf
from torch.nn.functional import cosine_similarity
from typing import Dict
import operator
import torchvision

path_img_input="test/1/test_0002_aligned.jpg"
path_img_output="product/{path_img_input}.jpg"
path_config="gpu.config.yml"


cfg = OmegaConf.load(path_config)

# initialize
analyzer = FaceAnalyzer(cfg.analyzer)

# warmup
response = analyzer.run(
        path_image=path_img_input,
        batch_size=cfg.batch_size,
        fix_img_size=cfg.fix_img_size,
        return_img_data=False,
        include_tensors=True,
        path_output=path_img_output,
    )