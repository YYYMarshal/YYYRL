# 环境配置

```cmd
conda create -n YYYRL python=3.8
conda activate YYYRL
pip install gym==0.19.0
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install tyro
pip install "cython<3"
pip install glfw
pip install imageio
pip install matplotlib

# 2024-8-15 19:37:04
# env.render() 时出错，解决方案：
pip install pyglet==1.5.0
```

并将 `mujoco_py` 文件夹放入 `S:\Users\YYYXB\anaconda3\envs\YYYRL\Lib\site-packages` 中。

