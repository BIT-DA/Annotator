# Step-by-Step Installation

This codebase is tested with `python=3.10`, `torch==1.11.0` and `torchvision==0.12.0`, `CUDA 11.3`.


### Step 1: Create Enviroment
```Shell
conda create -n annotator -y python=3.10
conda activate annotator
```

### Step 2: Install Packages
```Shell
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch -c conda-forge

conda install google-sparsehash -c bioconda

pip install pyyaml easydict numba wandb setproctitle prettytable sharedarray tqdm
```

### Step 3: Install Necessary Libraries

#### 3.1 - [TorchSparse](https://github.com/mit-han-lab/torchsparse)
**Note:** The following steps are **required** in order to use the `voxel` and `fusion` backbones in this codebase.

- Make a directory named `torchsparse_dir`
```Shell
cd package/
mkdir torchsparse_dir/
```

- Unzip the `.zip` files in `package/`
```Shell
unzip sparsehash.zip
unzip torchsparse.zip
```

- Setup `sparsehash` (Note that `${ROOT}` should be your home path to the `Annotator` folder)
```Shell
cd sparsehash/
./configure --prefix=/${ROOT}/Annotator/package/torchsparse_dir/sphash/

make

make install
```

- Compile `torchsparse`
```Shell
cd ..
pip install ./torchsparse
```

- It takes a while to build wheels. After successfully building `torchsparse`, you should see the following:
```Shell
Processing ./torchsparse
  Preparing metadata (setup.py) ... done
Building wheels for collected packages: torchsparse
  Building wheel for torchsparse (setup.py) ... done
  Created wheel for torchsparse: filename=torchsparse-2.0.0b0-cp310-cp310-linux_x86_64.whl size=8113060 sha256=aa5442e7d7b4537b7b18580ba5bd32c1fcb4930c3e0e46c811d4d40275e22610
  Stored in directory: /tmp/pip-ephem-wheel-cache-74ng7icc/wheels/51/5d/42/779ce27f2607ea50a81bd455bbb914023ea7f45a54e2174e0f
Successfully built torchsparse
Installing collected packages: torchsparse
Successfully installed torchsparse-2.0.0b0
```



#### 3.2 - [PyTorch Scatter](https://github.com/rusty1s/pytorch_scatter)

```Shell

conda install pytorch-scatter -c pyg
```



#### 3.3 - [nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit)
Note: This toolkit is required in order to run experiments on the [nuScenes](https://www.nuscenes.org/nuscenes) dataset.
```shell
pip install nuscenes-devkit 
```


