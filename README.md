# awesome-synthetic-data （ Synthetic Data 指南及实用资源 ）

## 入门与教程

### 博客与文章
- [RNN 的惊人有效性](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) - Andrej Karpathy 对循环神经网络（RNN）的介绍。
- [Annotated Diffusion](https://huggingface.co/blog/annotated-diffusion) - Diffusion 模型原论文的教程附代码。

### 视频与课程
- [生成数据：通过估计数据分布的梯度学习](https://youtu.be/nv-WTeKRLl0) - Stanford 的 Yang Song 视频，讲解理论及应用。

## 开源库

### 文本、表格与时间序列数据
- [gretel-synthetics](https://github.com/gretelai/gretel-synthetics) - 生成结构化和非结构化文本、表格以及多变量时间序列数据，支持差分隐私。
- [SDV](https://github.com/sdv-dev/SDV) - 表格、关系型和时间序列数据的合成器。
- [ydata-synthetic](https://github.com/ydataai/ydata-synthetic) - 生成结构化数据的工具。

### 图像
- [StyleGAN 3](https://github.com/NVlabs/stylegan3) - NeurIPS 2021 提出的高质量图像生成模型。
- [Denoising Diffusion Pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) - 扩散模型的 PyTorch 实现。

### 音频
- [Jukebox](https://github.com/openai/jukebox/) - OpenAI 提供的音乐生成模型。

### 仿真
- [AirSim](https://microsoft.github.io/AirSim/) - 用于无人机、汽车等仿真的工具。
- [Unity Perception](https://github.com/Unity-Technologies/com.unity.perception) - Unity 中的感知仿真工具包。

## 数据集资源
- [HuggingFace Datasets](https://huggingface.co/docs/datasets/index) - NLP、计算机视觉和音频任务的数据集。
- [Kaggle Datasets](https://www.kaggle.com/datasets) - 数据科学与机器学习的数据集。
- [Papers with Code - Datasets](https://paperswithcode.com/datasets) - 提供机器学习论文、代码和数据集的资源。

## 学术论文

### 生成对抗网络 (GANs)
- [生成对抗网络](https://arxiv.org/abs/1406.2661) - Ian Goodfellow 等人的经典论文。
- [条件生成对抗网络](https://arxiv.org/abs/1411.1784) - Mehdi Mirza 提出的条件 GAN 模型。
- [Wasserstein GAN](https://arxiv.org/abs/1701.07875) - 优化 GAN 的训练稳定性。
- [时间序列生成对抗网络](https://proceedings.neurips.cc/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf) - 用于生成时间序列数据的 GAN 方法。

### Diffusion 模型
- [通过估计数据分布的梯度进行生成建模](https://yang-song.github.io/blog/2021/score/) - Yang Song 论文及博客。
- [扩散模型是自编码器](https://benanne.github.io/2022/01/31/diffusion.html) - 深入理解扩散模型。
