# GUIDANCE-TGI

将 TGI 的推理加速集成到 guidance 中

## 快速开始

### 安装 guidance

```sh
git clone git@github.com:ZJUICI/guidance.git
cd guidance && python setup.py install
```

### 启动 TGI

[huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference)

如果本地有 GPU 条件, 建议使用 docker 的方式启动

[TGI docker start](https://github.com/huggingface/text-generation-inference#docker)

```sh
export volume=<your model path>

docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data zjuici/mirror.huggingface.text-generation-inference:1.1.0-guidance-stop --model-id $model
```

### 启动 guidance

* 配置环境变量

    ```sh
    export LOG_LEVEL=DEBUG # 部分日志
    export TGI_ENDPOINT_URL=http://127.0.0.1:8080/ # TGI的地址
    ```

* 打开 `notebooks/tgi_examples/demo.ipynb`, 运行 notebook

#### 注意

* `dsl_template.py`, `format_template.py` 需要在 `PATHPYTHON` 中,或者保证和需要运行的 `demo.ipynb` 同级,这俩文件目前被 `demo.ipynb` 所依赖
* 目前验证了一个 shape dsl 的输出, 其余的还未验证
