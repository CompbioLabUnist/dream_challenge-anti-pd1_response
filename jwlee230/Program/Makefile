# Variables
DATE := $(shell date "+%y%m%d")
RANDOM := $(shell bash -c 'echo $$RANDOM')
CPUS = 50
MEMS = 500G
PWD := $(shell pwd)
TOOLS = 
DOCKER = docker run --rm $(VOLUME_OPTS) $(RUN_OPTS) $(BASE_IMAGE)

# IMAGE_NAME
BASE_IMAGE = antipd1-jwlee230:latest
Q1_IMAGE = docker.synapse.org/syn23664597/antipd1-q1-model:$(DATE)
Q2_IMAGE = docker.synapse.org/syn23664597/antipd1-q2-model:$(DATE)
Q3_IMAGE = docker.synapse.org/syn23664597/antipd1-q3-model:$(DATE)

# Options
RUN_OPTS = --tty --cpus="$(CPUS)" --memory="$(MEMS)"
VOLUME_OPTS = --volume $(abspath Python):/Python --volume $(abspath Data):/Data:ro --volume $(abspath Output):/Output --volume $(abspath output):/output

# General
all:
.PHONY += all

log Output Tools output:
	mkdir $@

# Build Images
base.build: Dockerfile | Output output
	rm -fv $@
	docker images | grep $(BASE_IMAGE) && docker rmi $(BASE_IMAGE) || true
	docker build --rm --tag $(BASE_IMAGE) . | tee $@

q1.build: q1.Dockerfile base.build | Output output
	rm -fv $@
	docker images | grep $(Q1_IMAGE) && docker rmi $(Q1_IMAGE) || true
	docker build --rm --tag $(Q1_IMAGE) --file $< . | tee $@

q2.build: q2.Dockerfile base.build | Output output
	rm -fv $@
	docker images | grep $(Q2_IMAGE) && docker rmi $(Q2_IMAGE) || true
	docker build --rm --tag $(Q2_IMAGE) --file $< . | tee $@

q3.build: q3.Dockerfile base.build | Output output
	rm -fv $@
	docker images | grep $(Q3_IMAGE) && docker rmi $(Q3_IMAGE) || true
	docker build --rm --tag $(Q3_IMAGE) --file $< . | tee $@

build: base.build
.PHONY += build

build-all: base.build q1.build q2.build q3.build
.PHONY += build-all

# Interactive
interactive: base.build
	$(DOCKER) --interactive $(BASE_IMAGE) /bin/bash || true

q1-interactive: q1.build
	docker run --rm $(RUN_OPTS) $(Q1_IMAGE)
.PHONY += q1-interactive

q2-interactive: q2.build
	docker run --rm $(RUN_OPTS) $(Q2_IMAGE)
.PHONY += q2-interactive

q3-interactive: q3.build
	docker run --rm $(RUN_OPTS) $(Q3_IMAGE)
.PHONY += q3-interactive

# Delete Docker Images
delete: base.build
	docker rmi $(BASE_IMAGE)
	rm -fv $^
.PHONY += delete

q1-delete: q1.build
	docker rmi $(Q1_IMAGE)
	rm -fv $^
.PHONY += q1-delete

q2-delete: q2.build
	docker rmi $(Q2_IMAGE)
	rm -fv $^
.PHONY += q2-delete

q3-delete: q3.build
	docker rmi $(Q3_IMAGE)
	rm -fv $^
.PHONY += q3-delete

delete-all: delete q1-delete q2-delete q3-delete
.PHONY += delete-all

# Submit Docker
q1-submit: q1.build
	docker push $(Q1_IMAGE)
.PHONY += q1-submit

q2-submit: q2.build
	docker push $(Q2_IMAGE)
.PHONY += q2-submit

q3-submit: q3.build
	docker push $(Q3_IMAGE)
.PHONY += q3-submit

# SGE
tmp.sh: base.build
	echo "make -C $(PWD) latest" > tmp.sh

run: tmp.sh | log Output
	qsub -cwd -l h_vmem=$(MEMS) -m abe -M "230@fumire.moe" -N Dream_$(DATE) -pe smp $(CPUS) -o $(abspath log) -e $(abspath log) $<
.PHONY += run

# Actual
latest: step01
.PHONY += latest

# Step 01 (Read data from TIDEpy)
Output/Step01:
	mkdir -p $@

Output/Step01/Gene_Ref.tar.gz: Python/step01.py Data/TIDEpy/Gene_Ref.pkl | Output/Step01 base.build
	$(DOCKER) python3 $(addprefix /,$^ $@)

Output/Step01/model.tar.gz: Python/step01.py Data/TIDEpy/model.pkl | Output/Step01 base.build
	$(DOCKER) python3 $(addprefix /,$^ $@)

step01: Output/Step01/Gene_Ref.tar.gz
.PHONY += step01

# Step 02 (Clearify & Merge data)
Output/Step02:
	mkdir -p $@

Output/Step02/merged.tar.gz: Python/step02.py Data/expression_final.tsv Data/exp_TPM_final.tsv Data/Final_clin.tsv | Output/Step02 base.build
	$(DOCKER) python3 $(addprefix /,$^ $@)

step02: Output/Step02/merged.tar.gz
.PHONY += step02

# Step 03 (Make t-SNE)
Output/Step03:
	mkdir -p $@

Output/Step03/merged.png: Python/step03.py Output/Step02/merged.tar.gz | Output/Step03 base.build
	$(DOCKER) python3 $(addprefix /,$^ $@) --cpus $(CPUS)

step03: Output/Step03/merged.png
.PHONY += step03

# Step 04 (Clearify & Merge Synapse Data)
Output/Step04:
	mkdir -p $@

Output/Step04/Synapse.tar.gz: Python/step04.py Data/SynapseData/clinical_data.csv $(sort $(wildcard Data/SynapseData/GRCh37ERCC*.csv)) | Output/Step04 base.build
	$(DOCKER) python3 $(addprefix /,$^ $@)

step04: Output/Step04/Synapse.tar.gz
.PHONY += step04

# Step 05 (Make t-SNE with Synapse data)
Output/Step05:
	mkdir -p $@

Output/Step05/Synapse.png: Python/step05.py Output/Step04/Synapse.tar.gz | Output/Step05 base.build
	$(DOCKER) python3 $(addprefix /,$^ $@) --cpus $(CPUS)

step05: Output/Step05/Synapse.png
.PHONY += step05
