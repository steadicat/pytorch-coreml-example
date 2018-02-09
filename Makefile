VIRTUALENV:=$(shell which virtualenv)
ENV=env
SITE_PACKAGES=$(ENV)/lib/python2.7/site-packages
PYTHON=/usr/bin/python
LOAD_ENV=source $(ENV)/bin/activate

PYTORCH=http://download.pytorch.org/whl/torch-0.3.0.post4-cp27-none-macosx_10_6_x86_64.whl
ONNX_COREML=git+https://github.com/onnx/onnx-coreml.git
TNT=git+https://github.com/pytorch/tnt.git

dev:
	cd test-ui && ./node_modules/.bin/parcel index.html & open http://localhost:1234/
.PHONY: dev

env: $(VIRTUALENV)
	virtualenv env --python=$(PYTHON)

$(SITE_PACKAGES)/torch:
	$(LOAD_ENV) && pip install $(PYTORCH)

$(SITE_PACKAGES)/onnx_coreml:
	$(LOAD_ENV) && pip install $(ONNX_COREML)

$(SITE_PACKAGES)/torchnet:
	$(LOAD_ENV) && pip install $(TNT)

SplitModel.mlmodel: env $(SITE_PACKAGES)/torch $(SITE_PACKAGES)/onnx_coreml $(SITE_PACKAGES)/torchnet train.py data.json
	$(LOAD_ENV) && python train.py

train:
	@touch data.json
	@make SplitModel.mlmodel
.PHONY: train

prediction.json: SplitModel.mlmodel
	$(LOAD_ENV) && python predict.py

predict: prediction.json
.PHONY: predict