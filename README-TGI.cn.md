## GUIDANCE-TGI
将tgi的推理加速集成到guidance中

## 快速开始
### 启动tgi
[huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference)

如果本地有GPU条件,建议使用docker的方式启动

[tgi docker start](https://github.com/huggingface/text-generation-inference#docker)

```bash
export volume=<your model path>

docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data zjuici/mirror.huggingface.text-generation-inference:1.1.0-guidance-stop --model-id $model
```

### 启动guidance
#### 环境变量
```bash 
export LOG_LEVEL=DEBUG # 部分日志
export ENDPOINTS_URL=http://127.0.0.1:8080/ # TGI的地址
```
* clone github repo use `git clone `
* 安装依赖,可以使用[pdm](https://github.com/pdm-project/pdm)管理,也提供了`requirements.txt`
* 打开`demo.ipynb`,运行notebook


#### 注意
* `dsl_template.py`, `format_template.py` 需要在`PATHPYTHON`中,或者保证和需要运行的`demo.ipynb`同级,这俩文件目前被`demo.ipynb`所依赖. **后续会做优化修改**
* 目前验证了一个shape dsl的输出,其余的还未验证

