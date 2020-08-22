# If a Python interpreter is specified, then creates a virtualenv from it
# If empty string is given, chainer is not installed. Note that ESPnet doesn't support any versions except for chainer=6.0.0
CHAINER_VERSION := 6.0.0
# Disable cupy installation
NO_CUPY :=
# PyTorch version: 0.4.1, 1.0.0, 1.0.1, 1.1.0, 1.2.0, 1.3.0, 1.3.1, 1.4.0, 1.5.0, 1.5.1, 1.6.0
TH_VERSION := 1.4.0
WGET := wget --tries=3

<<<<<<< HEAD
# Both Miniconda2/3 can install any Python versions
#CONDA_URL := https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
CONDA_URL := https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh         # Mac
# PyTorch>=1.0.0 requires gcc>=4.9 when buliding the extensions
GCC_VERSION := $(shell gcc -dumpversion)
=======
# Use pip for pytorch installation even if you have anaconda
ifneq ($(shell test -f ./activate_python.sh && grep 'conda activate' ./activate_python.sh),)
IS_CONDA := 0
USE_PIP :=
else
IS_CONDA :=
USE_PIP := 0
endif
>>>>>>> upstream/master


# Set if install binaries on CPU mode e.g. make CPU_ONLY=1
# If you don't have any GPUs, this value will be set automatically
ifeq ($(shell which nvcc),) # 'nvcc' not found
CPU_ONLY := 0
else
CPU_ONLY :=
endif

ifeq ($(strip $(CPU_ONLY)),)
# Derive CUDA version from nvcc
CUDA_VERSION = $(shell nvcc --version | grep "Cuda compilation tools" | cut -d" " -f5 | sed s/,//)
CUDA_VERSION_WITHOUT_DOT = $(strip $(subst .,,$(CUDA_VERSION)))
CONDA_PYTORCH := pytorch=$(TH_VERSION) cudatoolkit=$(CUDA_VERSION)
PIP_PYTORCH := torch==$(TH_VERSION) -f https://download.pytorch.org/whl/cu$(CUDA_VERSION_WITHOUT_DOT)/torch_stable.html

else
CONDA_PYTORCH := pytorch=$(TH_VERSION) cpuonly
PIP_PYTORCH := torch==$(TH_VERSION) -f https://download.pytorch.org/whl/cpu/torch_stable.html
NO_CUPY := 0
endif

ifeq ($(shell expr $(CUDA_VERSION_WITHOUT_DOT) \>= 102), 1)
# cupy==6.0.0 doesn't support CUDA=10.2 or later
NO_CUPY := 0
endif
PIP_CHAINER := chainer==$(CHAINER_VERSION)
<<<<<<< HEAD
else
<<<<<<< HEAD
CONDA_PYTORCH := pytorch=$(TH_VERSION)
CUDA_DEPS :=
=======
PIP_CHAINER := chainer==$(CHAINER_VERSION) cupy-cuda$(CUDA_VERSION_WITHOUT_DOT)==$(CHAINER_VERSION)
>>>>>>> upstream/master
endif
=======
>>>>>>> upstream/master


.PHONY: all clean

all: kaldi showenv python conda_packages.done sctk.done sph2pipe.done check_install

ifneq ($(strip $(CHAINER_VERSION)),)
python: activate_python.sh warp-ctc.done warp-transducer.done espnet.done pytorch.done chainer_ctc.done chainer.done
else
python: activate_python.sh warp-ctc.done warp-transducer.done espnet.done pytorch.done
endif

extra: nkf.done moses.done mwerSegmenter.done pesq kenlm.done pyopenjtalk.done

kaldi:
	test -f kaldi/egs/wsj/s5/utils/parse_options.sh || { echo -e "Error: Put Kaldi here!\n    $$ ln -s <kaldi-root> kaldi"; exit 1; }

activate_python.sh:
	test -f activate_python.sh || { echo "Error: Run ./setup_python.sh or ./setup_anaconda.sh"; exit 1; }

################ Logging ################
showenv: activate_python.sh
ifeq ($(strip $(CPU_ONLY)),)
	[ -n "${CUDA_HOME}" ] || { echo -e "Error: CUDA_HOME is not set.\n    $$ . ./setup_cuda_env.sh <cuda-root>"; exit 1; }
	@echo CUDA_VERSION=$(CUDA_VERSION)
	. ./activate_python.sh && python3 check_pytorch_cuda_compatibility.py $(TH_VERSION) ${CUDA_VERSION} || { echo "Change TH_VERSION or CUDA_VERSION"; exit 1; }
else
	@echo Perform on CPU mode: CPU_ONLY=$(CPU_ONLY)
endif
	@echo PYTHON=$(shell . ./activate_python.sh && command -v python3)
	@echo PYTHON_VERSION=$(shell . ./activate_python.sh && python3 --version)
	@echo IS_CONDA=$(IS_CONDA)
	@echo USE_PIP=$(USE_PIP)
	@echo TH_VERSION=$(TH_VERSION)
	@echo CONDA_PYTORCH=$(CONDA_PYTORCH)
	@echo PIP_PYTORCH=$(PIP_PYTORCH)
	@echo CHAINER_VERSION=$(CHAINER_VERSION)
	@echo PIP_CHAINER=$(PIP_CHAINER)
	@echo NO_CUPY=$(NO_CUPY)

#########################################

cmake.done: activate_python.sh
	. ./activate_python.sh && { command -v cmake || conda install -y cmake; }
	touch cmake.done
flac.done: activate_python.sh
	. ./activate_python.sh && { command -v flac || conda install -y libflac -c conda-forge; }
	touch flac.done
ffmpeg.done: activate_python.sh
	. ./activate_python.sh && { command -v ffmpeg || conda install -y ffmpeg -c conda-forge; }
	touch ffmpeg.done
sox.done: activate_python.sh
	. ./activate_python.sh && { command -v sox || conda install -y sox -c conda-forge; }
	touch sox.done
sndfile.done: activate_python.sh
	. ./activate_python.sh && { python3 -c "from ctypes.util import find_library as F; assert F('sndfile') is not None" || conda install -y libsndfile -c conda-forge; }
	touch sndfile.done
ifneq ($(strip $(IS_CONDA)),)
conda_packages.done: cmake.done flac.done ffmpeg.done sox.done sndfile.done
else
conda_packages.done:
endif
	touch conda_packages.done


sctk.done:
	./installers/install_sctk.sh
	touch sctk.done
sph2pipe.done:
	./installers/install_sph2pipe.sh
	touch sph2pipe.done

<<<<<<< HEAD
ifneq ($(strip $(PYTHON)),)
activate_python.sh:
	test -d venv || $(PYTHON) -m venv venv
	echo ". $(CURDIR)/venv/bin/activate" > activate_python.sh
else
<<<<<<< HEAD
miniconda.sh:
	test -f miniconda.sh || $(WGET) $(CONDA_URL) -O miniconda.sh --no-check-certificate
venv: miniconda.sh
	test -d $(PWD)/venv || bash miniconda.sh -b -p $(PWD)/venv
	. venv/bin/activate && conda install -y setuptools -c anaconda
	. venv/bin/activate && conda install -y pip -c anaconda
	. venv/bin/activate && conda update -y conda
	. venv/bin/activate && conda install -y python=$(PYTHON_VERSION)
	. venv/bin/activate && conda info -a
espnet.done: venv
	. venv/bin/activate && conda install -y $(CONDA_PYTORCH) -c pytorch
	. venv/bin/activate && pip install -e ..
	touch espnet.done
=======
activate_python.sh:
	./installers/install_anaconda.sh "$(CONDA_ENV_NAME)" $(PYTHON_VERSION) $(CONDA)
	echo 'if [ -z "$${PS1:-}" ]; then' > activate_python.sh.bak
	echo '    PS1=__dummy__' >> activate_python.sh.bak
	echo 'fi' >> activate_python.sh.bak
	echo ". $(abspath $(CONDA))/etc/profile.d/conda.sh && conda deactivate && conda activate $(CONDA_ENV_NAME)" >> activate_python.sh.bak
	mv activate_python.sh.bak activate_python.sh
>>>>>>> upstream/master
endif
=======
>>>>>>> upstream/master

ifneq ($(strip $(USE_PIP)),)
pytorch.done: activate_python.sh
	. ./activate_python.sh && python3 -m pip install numpy --upgrade  # Workaround for "ImportError: numpy.core.multiarray failed to import"
	. ./activate_python.sh && python3 -m pip install $(PIP_PYTORCH)
	touch pytorch.done
else
pytorch.done: activate_python.sh
	. ./activate_python.sh && conda install -y $(CONDA_PYTORCH) -c pytorch
	touch pytorch.done
endif

<<<<<<< HEAD
<<<<<<< HEAD
warp-ctc.done: espnet.done # Manually installed
	touch warp-ctc.done

warp-transducer.done: espnet.done  # Manually installed
	touch warp-transducer.done

chainer_ctc.done: espnet.done
	. venv/bin/activate; pip install cython
	. venv/bin/activate; cd chainer_ctc && chmod +x install_warp-ctc.sh && ./install_warp-ctc.sh ; true
	. venv/bin/activate; cd chainer_ctc && pip install .
	touch chainer_ctc.done

nkf.done:
	rm -rf nkf
	mkdir -p nkf
	cd nkf; $(WGET) https://ja.osdn.net/dl/nkf/nkf-2.1.4.tar.gz --no-check-certificate
	cd nkf; tar zxvf nkf-2.1.4.tar.gz; cd nkf-2.1.4; $(MAKE) prefix=.
=======
espnet.done: pytorch.done
	. ./activate_python.sh && pip install -e ..  # Install editable mode by default
=======
# NOTE(kamo): conda_packages is not necessary for installation of espnet, but add it the dependencies just in case.
espnet.done: pytorch.done conda_packages.done
	. ./activate_python.sh && python3 -m pip install -e ..  # Install editable mode by default
>>>>>>> upstream/master
	touch espnet.done

chainer.done: espnet.done
ifneq ($(strip $(NO_CUPY)),)
	. ./activate_python.sh && python3 -m pip install $(PIP_CHAINER)
else
	# Precompiled cupy==6.0.0 for python>=3.8 is not provided
	. ./activate_python.sh && \
		if python3 -c "import sys; from distutils.version import LooseVersion as L; assert L(sys.version) < L('3.8')" 2>&1 /dev/null; then \
			python3 -m pip install $(PIP_CHAINER) cupy-cuda$(CUDA_VERSION_WITHOUT_DOT)==$(CHAINER_VERSION); \
		else \
			python3 -m pip install $(PIP_CHAINER) cupy==$(CHAINER_VERSION); \
		fi
endif
	touch chainer.done

# NOTE(kamo): Add conda_packages.done if cmake is used
warp-ctc.done: pytorch.done conda_packages.done
	. ./activate_python.sh && ./installers/install_warp-ctc.sh
	touch warp-ctc.done

warp-transducer.done: pytorch.done conda_packages.done
	. ./activate_python.sh && ./installers/install_warp-transducer.sh
	touch warp-transducer.done

chainer_ctc.done: chainer.done
	. ./activate_python.sh && ./installers/install_chainer_ctc.sh
	touch chainer_ctc.done

nkf.done:
	./installers/install_nkf.sh
>>>>>>> upstream/master
	touch nkf.done

pyopenjtalk.done: espnet.done conda_packages.done
	. ./activate_python.sh && ./installers/install_pyopenjtalk.sh
	touch pyopenjtalk.done

moses.done:
	git clone --depth 1 https://github.com/moses-smt/mosesdecoder.git moses
	touch moses.done

mwerSegmenter.done:
<<<<<<< HEAD
	$(WGET) https://www-i6.informatik.rwth-aachen.de/web/Software/mwerSegmenter.tar.gz --no-check-certificate
	tar zxvf mwerSegmenter.tar.gz
	rm mwerSegmenter.tar.gz
=======
	./installers/install_mwerSegmenter.sh
>>>>>>> upstream/master
	touch mwerSegmenter.done

kenlm.done: espnet.done conda_packages.done
	. ./activate_python.sh && ./installers/install_kenlm.sh
	touch kenlm.done

pesq: PESQ/P862_annex_A_2005_CD/source/PESQ
PESQ/P862_annex_A_2005_CD/source/PESQ:
	./installers/install_pesq.sh


check_install: python
ifeq ($(strip $(CPU_ONLY)),)

ifneq ($(strip $(NO_CUPY)),)
	. ./activate_python.sh; python3 check_install.py --no-cupy
else
	. ./activate_python.sh; python3 check_install.py
endif

else

ifneq ($(strip $(NO_CUPY)),)
	. ./activate_python.sh; python3 check_install.py --no-cuda --no-cupy
else
	. ./activate_python.sh; python3 check_install.py --no-cuda
endif

<<<<<<< HEAD
pesq: PESQ/P862_annex_A_2005_CD/source/PESQ
PESQ/P862_annex_A_2005_CD/source/PESQ: PESQ
	cd PESQ/P862_annex_A_2005_CD/source; gcc *.c -lm -o PESQ
PESQ: PESQ.zip
	mkdir -p PESQ_P.862.2
	unzip PESQ.zip -d PESQ_P.862.2
	unzip "PESQ_P.862.2/Software/P862_annex_A_2005_CD  wav final.zip" -d PESQ_P.862.2
	rm -rf PESQ
	ln -s PESQ_P.862.2 PESQ
PESQ.zip:
	$(WGET) 'http://www.itu.int/rec/dologin_pub.asp?lang=e&id=T-REC-P.862-200511-I!Amd2!SOFT-ZST-E&type=items' -O PESQ.zip --no-check-certificate
=======
endif
>>>>>>> upstream/master


clean: clean_extra
	rm -rf warp-ctc warp-transducer chainer_ctc
	rm -rf *.done
	find . -iname "*.pyc" -delete

clean_python:
	rm -rf warp-ctc warp-transducer chainer_ctc
	rm -f warp-ctc.done chainer_ctc.done espnet.done chainer.done pytorch.done warp-transducer.done
	find . -iname "*.pyc" -delete

clean_extra:
	rm -rf nkf.done swig.done moses.done mwerSegmenter.done
	rm -rf hts_engine_API.done open_jtalk.done pyopenjtalk.done
	rm -rf nkf mecab swig moses mwerSegmenter
	rm -rf PESQ PESQ.zip
