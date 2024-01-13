# Wildfire hazardous prediction of buildings with XAI.
This is a PyTorch Implementation of "Wildfire hazardous prediction of buildings with XAI"  project.
This work uses TabNet architecture for 


## Our dataset
- You can find our dataset in '/data' folder

## Methods Used
- TabNet
- 1D-CNN
- Two-stage Nerual Network
- Three-stage Neural Network
- PCA
- Cross Validation

## Performances

<table style="margin: auto">
  <tr>
    <th>model</th>
    <th># of<br />params[M]</th>
    <th>Feature Size</th>
    <th>Accuracy</th>
  </tr>
  <tr>
    <td>TabNet</td>
    <td align="right">4.7</td>
    <td align="right">512</td>
    <td align="right">85.73</td>
  </tr>
</table>


## How to use

### Library 설치
```
conda create -n test python=3.8 -y
conda activate test
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install pandas matplotlib scikit-learn
```

### Training
```
sh train.sh
```

### Evaluation
```
sh disc_eval.sh
```

## Citation
```
@article{juntae2023,
  title={Wildfire hazardous prediction of buildings with XAI},
  author={Juntae Kim},
  journal={},
  year={2024}
}
