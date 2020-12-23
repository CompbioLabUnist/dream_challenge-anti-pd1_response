# Variables
DATE := $(shell date "+%y%m%d")
RANDOM := $(shell bash -c 'echo $$RANDOM')
CPUS = 10
MEMS = 100G
PWD := $(shell pwd)
TOOLS = 
DOCKER = docker run --rm $(VOLUME_OPTS) $(RUN_OPTS)

# IMAGE_NAME
BASE_IMAGE = antipd1-jwlee230:latest
Q1_IMAGE = docker.synapse.org/syn23664597/awesome-antipd1-q1-model:$(DATE)
Q2_IMAGE = docker.synapse.org/syn23664597/awesome-antipd1-q2-model:$(DATE)
Q3_IMAGE = docker.synapse.org/syn23664597/awesome-antipd1-q3-model:$(DATE)

# Options
RUN_OPTS = --tty --cpus="$(CPUS)" --memory="$(MEMS)"
VOLUME_OPTS = --volume $(abspath Python):/Python --volume $(abspath Data):/Data:ro --volume $(abspath Output):/Output

# General
all:
.PHONY += all

log Output Tools:
	mkdir $@

# Build Images
base.build: Dockerfile | Output
	rm -fv $@
	docker images | grep $(BASE_IMAGE) && docker rmi $(BASE_IMAGE) || true
	docker build --rm --tag $(BASE_IMAGE) . | tee $@

q1.build: q1.Dockerfile base.build | Output
	rm -fv $@
	docker images | grep $(Q1_IMAGE) && docker rmi $(Q1_IMAGE) || true
	docker build --rm --tag $(Q1_IMAGE) --file $< . | tee $@

q2.build: q2.Dockerfile base.build | Output
	rm -fv $@
	docker images | grep $(Q2_IMAGE) && docker rmi $(Q2_IMAGE) || true
	docker build --rm --tag $(Q2_IMAGE) --file $< . | tee $@

q3.build: q3.Dockerfile base.build | Output
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

delete-q1: q1.build
	docker rmi $(Q1_IMAGE)
	rm -fv $^
.PHONY += delete-q1

delete-q2: q2.build
	docker rmi $(Q2_IMAGE)
	rm -fv $^
.PHONY += delete-q2

delete-q3: q3.build
	docker rmi $(Q3_IMAGE)
	rm -fv $^
.PHONY += delete-q3

delete-all: delete delete-q1 delete-q2 delete-q3
.PHONY += delete-all

# Submit Docker
q1-submit: q1.build
	docker push $(Q1_IMAGE)

# SGE
tmp.sh: base.build
	echo "make -C $(PWD) latest" > tmp.sh

run: tmp.sh | log Output
	qsub -cwd -l h_vmem=$(MEMS) -m abe -M "230@fumire.moe" -N Dream_$(DATE) -pe smp $(CPUS) -o $(abspath log) -e $(abspath log) $<
.PHONY += run

# Actual
latest: step01
.PHONY += latest
