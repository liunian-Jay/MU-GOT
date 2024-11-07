
## 本项目基于MinerU和GOT-OCR2.0 实现pdf解析
### 本项目仅为学习交流, 欢迎大家对不合适的地方改进
请大家关注[GOT](https://github.com/Ucas-HaoranWei/GOT-OCR2.0),[MinerU](https://github.com/opendatalab/MinerU)

🔥 2024/11/7 debug 修复了batch推理不说人话，vllm batch推理吐token的速度自测几个case可以提升数倍

+ 本项目主要修改:
  + vllm 0.5.3版本实现了GOT加速
  + 去除了MinerU的本地文件存储，直接用变量传递
- 本项目主要流程:
  - 基于MinerU实现了pdf到markdown的解析，这一步未进行表格识别(其自带的表格识别,但是速度太慢了)
  - 基于GOT进一步对每个表格进行识别，最终处理得到文本形式的


### 运行
适当安装依赖,主要为torch 2.3.1, vllm 0.5.3, transformer </br>
或直接安装新环境```conda env create -f environment.yml```
### 项目安装：
进入项目根目录，执行 ```pip install -e .``` 安装, 其会自动安装MinerU和GOT-OCR2.0所依赖的库
#### 需要把PDF_parsing/magic-pdf.json的模型路径替换成你的路径,PDF_parsing/__init__.py需同样替换


### 尚待改进
- ~~vllm版本的GOT 输入为batch时候生成的不说人话~~
- 最后生成的text不是纯markdown, GOT-OCR2.0将表格转为了latex, 所以最后生成的是markdown格式和latex表格的混合

