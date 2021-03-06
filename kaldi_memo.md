### Kald Install
    - Kaldi installation
        - GPU Version :  Ubuntu 16.04  Cuda = 8.0 or 10.0

    - Install MKL

    Setting mkl-root to your installation location (eg. ~/Intel/mkl/)
    Set mkl.h header file:
    export CPLUS_INCLUDE_PATH="/home/luoxuan/intel/compilers_and_libraries_2019.5.281/linux/mkl/include"

    - Install protobuf

    sudo apt-get install autoconf automake libtool curl make g++ unzip -y
    git clone https://github.com/google/protobuf.git
    cd protobuf
    git submodule update --init --recursive
    ./autogen.sh
    make
    make check
    sudo make install
    sudo ldconfig

    - Install other  -> warpctc_pytorch (???)

    - Compile Kaldi
    Check kaldi dependency by  ./check_dependencies.sh  in kaldi/tools/extras
    Install kaldi
        •	kaldi/tools/extras, run ./check_dependencies.sh and check that everything is okay
        •	kaldi/tools, run make OPENFST_VERSION=1.6.9 -j 8
        •	kaldi/src, run ./configure --shared && make depend && make -j 8
    =>./configure --shared --mkl-root=/home/luoxuan/intel --mkl-libdir=/home/luoxuan/intel/compilers_and_libraries_2019.5.281/linux/mkl/lib/intel64_lin/
        •	kaldi/tools, run ./install_srilm.sh (optional)

    - Install espnet


### Install Kaldi CPU
- Compile Kaldi
    - add --no-certificate-security
- Python Environment
    - Change mac-version miniconda.sh, not linux
    - Change to espnet-conda env, instead of default conda
    - build warpctc-pytorch10-cpu, instead of pip install, cause no such
    - Install llvm(clang++) libomp to compile since fopenmp not supported in Mac gcc4.2

make CUPY_VERSION='' PYTHON_VERSION=3.7 TH_VERSION=1.2.0
![unsupported option '-fopenmp'](https://stackoverflow.com/questions/43555410/enable-openmp-support-in-clang-in-mac-os-x-sierra-mojave)

cmake -DCMAKE_C_COMPILER="/usr/local/opt/llvm/bin/clang" -DCMAKE_CXX_COMPILER="/usr/local/opt/llvm/bin/clang++" ..


### Install GPU
- install mkl (full package)
https://codeyarns.com/2019/05/14/how-to-install-intel-mkl/



### Kaldi architect

    In egs/**/s5/
    cmd.sh                     # 并行执行命令，通常分 run.pl, queue.pl 两种
    config                       # 参数定制化配置文件， mfcc, decode, cmvn 等配置文件
    local                        # 工程定制化内容
    path.sh                    # 环境变量相关脚本
    run.sh                      # 整体流程控制脚本，主入口脚本
    steps                       # 存放单一步骤执行的脚本
    utils                         # 存放解析文件，预处理等相关工具的脚本

要介绍了两种方式执行：
    常见用法：
             run.pl some.log a b c
        即在 bash 环境中执行 a b c 命令，并将日志输出到 some.log 文件中
    并行任务：
             run.pl JOB=1:4 some.JOB.log  a b c JOB
        即在 bash 环境中执行 a b c JOB 命令，并将日志输出到 some.JOB.log 文件中, 其中 JOB 表示执行任务的名称， 任意一个 Job 失败，整体失败。

    GPU
          run.pl --ngpu 3 some.log a
    run.pl some.log my-prog "--opt=foo bar" foo \|  other-prog baz
    and run.pl will run something like:
    ( my-prog '--opt=foo bar' foo |  other-prog baz ) >& some.log
