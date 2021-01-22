# Variables
DATE := $(shell date "+%y%m%d")
RANDOM := $(shell bash -c 'echo $$RANDOM')
CPUS = 10
MEMS = 100G
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
VOLUME_OPTS = --volume $(abspath Python):/Python --volume $(abspath Data):/Data:ro --volume $(abspath data):/data:ro --volume $(abspath Output):/Output --volume $(abspath output):/output

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

q1.build: q1.Dockerfile base.build Python/q1-final.py | Output output
	rm -fv $@
	docker images | grep $(Q1_IMAGE) && docker rmi $(Q1_IMAGE) || true
	docker build --rm --tag $(Q1_IMAGE) --file $< . | tee $@

q2.build: q2.Dockerfile base.build Python/q2-final.py | Output output
	rm -fv $@
	docker images | grep $(Q2_IMAGE) && docker rmi $(Q2_IMAGE) || true
	docker build --rm --tag $(Q2_IMAGE) --file $< . | tee $@

q3.build: q3.Dockerfile base.build Python/q3-final.py | Output output
	rm -fv $@
	docker images | grep $(Q3_IMAGE) && docker rmi $(Q3_IMAGE) || true
	docker build --rm --tag $(Q3_IMAGE) --file $< . | tee $@

build: base.build
.PHONY += build

build-all: base.build q1.build q2.build q3.build
.PHONY += build-all

# Interactive
interactive: base.build
	docker run --rm --interactive $(RUN_OPTS) $(BASE_IMAGE) /bin/bash || true

q1-interactive: q1.build Output/Step11/PFS.RF.tar.gz
	docker run --rm $(RUN_OPTS) $(VOLUME_OPTS) $(Q1_IMAGE)
.PHONY += q1-interactive

q2-interactive: q2.build Output/Step11/OS.RF.tar.gz
	docker run --rm $(RUN_OPTS) $(VOLUME_OPTS) $(Q2_IMAGE)
.PHONY += q2-interactive

q3-interactive: q3.build
	docker run --rm $(RUN_OPTS) $(VOLUME_OPTS) $(Q3_IMAGE)
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
q1-submit: Python/q1-final.py Output/Step11/PFS.RF.tar.gz q1.build
	docker push $(Q1_IMAGE)
.PHONY += q1-submit

q2-submit: Python/q2-final.py Output/Step11/OS.RF.tar.gz q2.build
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
latest: step11
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

Output/Step02/merged.tar.gz: Python/step02.py Data/expression_final.tsv Data/exp_TPM_final.tsv Data/Final_clin_JW.tsv | Output/Step02 base.build
	$(DOCKER) python3 $(addprefix /,$^ $@)

step02: Output/Step02/merged.tar.gz
.PHONY += step02

# Step 03 (Make t-SNE)
Output/Step03:
	mkdir -p $@

Output/Step03/%.png: Python/step03.py Output/Step02/merged.tar.gz | Output/Step03 base.build
	$(DOCKER) python3 $(addprefix /,$^ $@) --cpus $(CPUS)

step03: Output/Step03/TMB.png Output/Step03/IHC.png
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

Output/Step05/%.Synapse.png: Python/step05.py Output/Step04/Synapse.tar.gz | Output/Step05 base.build
	$(DOCKER) python3 $(addprefix /,$^ $@) --cpus $(CPUS)

step05: Output/Step05/TMB.Synapse.png Output/Step05/PDL1.Synapse.png
.PHONY += step05

# Step 06 (Impute missing features)
Output/Step06:
	mkdir -p $@

Output/Step06/imputed.tar.gz: Python/step06.py Output/Step02/merged.tar.gz | Output/Step06 base.build
	$(DOCKER) python3 $(addprefix /,$^ $@)

Output/Step06/imputed.Synapse.tar.gz: Python/step06.py Output/Step04/Synapse.tar.gz | Output/Step06 base.build
	$(DOCKER) python3 $(addprefix /,$^ $@)

step06: Output/Step06/imputed.tar.gz Output/Step06/imputed.Synapse.tar.gz
.PHONY += step06

# Step 07 (t-SNE with imputed data)
Output/Step07:
	mkdir -p $@

Output/Step07/%.tar: Python/step07.py Output/Step06/%.tar.gz | Output/Step07 base.build
	$(DOCKER) python3 $(addprefix /,$^ $@)

step07: Output/Step07/imputed.tar Output/Step07/imputed.Synapse.tar
.PHONY += step07

# Step 08 (get R2 score)
Output/Step08:
	mkdir -p $@

Output/Step08/%.ours.not_imputed.tar.gz: Python/step08.py Output/Step02/merged.tar.gz | Output/Step08 base.build
	$(DOCKER) python3 $(addprefix /,$^ $@) --cpus $(CPUS)

Output/Step08/%.ours.imputed.tar.gz: Python/step08.py Output/Step06/imputed.tar.gz | Output/Step08 base.build
	$(DOCKER) python3 $(addprefix /,$^ $@) --cpus $(CPUS)

Output/Step08/%.Synapse.not_imputed.tar.gz: Python/step08.py Output/Step04/Synapse.tar.gz | Output/Step08 base.build
	$(DOCKER) python3 $(addprefix /,$^ $@) --cpus $(CPUS)

Output/Step08/%.Synapse.imputed.tar.gz: Python/step08.py Output/Step06/imputed.Synapse.tar.gz | Output/Step08 base.build
	$(DOCKER) python3 $(addprefix /,$^ $@) --cpus $(CPUS)

step08: Output/Step08/TMB.ours.not_imputed.tar.gz Output/Step08/IHC.ours.not_imputed.tar.gz Output/Step08/PFS.ours.not_imputed.tar.gz Output/Step08/OS.ours.not_imputed.tar.gz Output/Step08/TMB.ours.imputed.tar.gz Output/Step08/IHC.ours.imputed.tar.gz Output/Step08/PFS.ours.imputed.tar.gz Output/Step08/OS.ours.imputed.tar.gz Output/Step08/TMB.Synapse.not_imputed.tar.gz Output/Step08/PDL1.Synapse.not_imputed.tar.gz Output/Step08/TMB.Synapse.imputed.tar.gz Output/Step08/PDL1.Synapse.imputed.tar.gz
.PHONY += step08

# Step 09 (t-SNE with R2-score data)
Output/Step09:
	mkdir -p $@

Output/Step09/%.png: Python/step09.py Output/Step08/%.tar.gz | Output/Step09 base.build
	$(DOCKER) python3 $(addprefix /,$^ $@) --cpus $(CPUS)

step09: Output/Step09/TMB.ours.not_imputed.png Output/Step09/IHC.ours.not_imputed.png Output/Step09/TMB.ours.imputed.png Output/Step09/IHC.ours.imputed.png Output/Step09/TMB.Synapse.not_imputed.png Output/Step09/PDL1.Synapse.not_imputed.png Output/Step09/TMB.Synapse.imputed.png Output/Step09/PDL1.Synapse.imputed.png
.PHONY += step09

# Step 10 (Select & Merge columns)
Output/Step10:
	mkdir -p $@

Output/Step10/TMB.selected.tar.gz: Python/step10.py Output/Step08/TMB.ours.imputed.tar.gz Output/Step08/TMB.Synapse.imputed.tar.gz | Output/Step10 base.build
	$(DOCKER) python3 $(addprefix /,$^ $@)

Output/Step10/PDL1.selected.tar.gz: Python/step10.py Output/Step08/IHC.ours.imputed.tar.gz Output/Step08/PDL1.Synapse.imputed.tar.gz | Output/Step10 base.build
	$(DOCKER) python3 $(addprefix /,$^ $@)

Output/Step10/TMB.not_imputed.selected.tar.gz: Python/step10.py Output/Step08/TMB.ours.not_imputed.tar.gz Output/Step08/TMB.Synapse.not_imputed.tar.gz | Output/Step10 base.build
	$(DOCKER) python3 $(addprefix /,$^ $@)

Output/Step10/PDL1.not_imputed.selected.tar.gz: Python/step10.py Output/Step08/IHC.ours.not_imputed.tar.gz Output/Step08/PDL1.Synapse.not_imputed.tar.gz | Output/Step10 base.build
	$(DOCKER) python3 $(addprefix /,$^ $@)

step10: Output/Step10/TMB.selected.tar.gz Output/Step10/PDL1.selected.tar.gz Output/Step10/TMB.not_imputed.selected.tar.gz Output/Step10/PDL1.not_imputed.selected.tar.gz
.PHONY += step10

# Step 11 (Build Random-Forest model)
Output/Step11:
	mkdir -p $@

Output/Step11/%.RF.tar.gz: Python/step11.py Output/Step08/%.ours.not_imputed.tar.gz Data/SynapseData/clinical_data.csv Data/SynapseData/GRCh37ERCC_ensembl75_isoforms_tpm.csv | Output/Step11 base.build
	$(DOCKER) python3 $(addprefix /,$^ $@) --cpus $(CPUS) 1> $@.stdout 2> $@.stderr

Output/Step11/%.RF.imputed.tar.gz: Python/step11.py Output/Step08/%.ours.imputed.tar.gz Data/SynapseData/clinical_data.csv Data/SynapseData/GRCh37ERCC_ensembl75_isoforms_tpm.csv | Output/Step11 base.build
	$(DOCKER) python3 $(addprefix /,$^ $@) --cpus $(CPUS) 1> $@.stdout 2> $@.stderr

step11: Output/Step11/PFS.RF.tar.gz Output/Step11/OS.RF.tar.gz Output/Step11/PFS.RF.imputed.tar.gz Output/Step11/OS.RF.imputed.tar.gz
.PHONY += step11
