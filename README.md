# SleevePose

![Pipeline](overall_pipeline.png)

SleevePose is a wearable system for continuous 3D hand pose estimation using an elastic pressure sensing sleeve. It captures forearm surface deformations via a 13×9 resistive textile matrix and predicts MANO hand mesh parameters through a ResNet+Transformer pipeline. The dataset includes 720k synchronized pressure and RGB frames from 18 users.

## Dataset
The dataset used in this project can be downloaded from the following link:
[Download Dataset](http://210.45.71.78:8888/d/54e3006c7a764f84a126/)

## Contact
If you have any questions, feel free to contact me via email: [nmt@mail.ustc.edu.cn](mailto:nmt@mail.ustc.edu.cn)

## Project Structure
Below is an overview of the project structure and the purpose of each file:

- **data_loader.py**: Contains the `HandPoseDataset` class for loading and preprocessing the pressure and label data.
- **model.py**: Implements the `ResNetTransformerModel` for predicting MANO hand mesh parameters, including the ResNet backbone and Transformer encoder.
- **trainer.py**: Includes the training loop, learning rate scheduler, and utility functions for training the model.
- **test.py**: Provides the testing pipeline for evaluating the trained model on the test dataset.
- **loss.py**: Defines the `HandParamsLoss` class for calculating various loss components, such as betas loss, hand pose loss, and keypoint loss.
- **model_evaluate.py**: Contains evaluation metrics and functions for validating the model's performance.
- **model_evalute_with_std.py**: Similar to `model_evaluate.py`, but includes additional evaluation with standard deviation metrics.
- **data/**: Directory containing the MANO model files and mean parameters.
  - **mano_mean_params.npz**: Precomputed mean parameters for the MANO model.
  - **mano/**: Directory containing the MANO model files.
- **geometry/**: Contains utility functions for geometric transformations.
  - **camera.py**: Functions for camera-related transformations.
  - **geometry.py**: Functions for geometric operations such as Rodrigues transformations.
- **README.md**: This file, providing an overview of the project.
- **LICENSE**: License file for the project.

## How to Run
### Prerequisites
- Python 3.8+
- PyTorch 1.10+
- Install the required dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Training
To train the model, run the following command:
```bash
python trainer.py
```

### Testing
To test the model, run:
```bash
python test.py
```

## Citation
If you use this code or dataset in your research, please cite our paper:
```
@article{SleevePose2026,
  title={SleevePose: Wearable System for Continuous 3D Hand Pose Estimation},
  author={Your Name},
  journal={Your Journal},
  year={2026}
}
```

---
Thank you for your interest in SleevePose!
