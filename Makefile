CLUSTER  ?= ai-inference
NS       ?= ai-inference
API_PORT ?= 8080
PY       ?= .venv/bin/python

.PHONY: venv lint test test-go test-py train build up down e2e kind-up kind-deploy kind-e2e kind-down bench

venv:
	python3 -m venv .venv && .venv/bin/pip install -r inference/requirements-dev.txt

lint:
	test -z "$$(gofmt -l api)" || (gofmt -l api; exit 1)
	cd api && go vet ./... && go run honnef.co/go/tools/cmd/staticcheck@v0.8.2 ./...
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .
	$(PY) -m mypy inference/app inference/tests benchmarks

test: test-go test-py

test-go:
	cd api && go vet ./... && go test -race ./...

test-py:
	$(PY) -m pytest inference/tests -q

train:
	cd inference && ../$(PY) -m app.train --out model

build:
	docker compose build

up: build
	API_PORT=$(API_PORT) docker compose up -d

down:
	docker compose down

e2e:
	tests/e2e.sh http://localhost:$(API_PORT)

kind-up:
	kind create cluster --config deployments/k8s/kind-config.yaml

kind-deploy: build
	kind load docker-image ai-inference-platform/api:dev ai-inference-platform/inference:dev --name $(CLUSTER)
	kubectl apply -k deployments/k8s
	kubectl -n $(NS) rollout restart deploy/api deploy/inference
	kubectl -n $(NS) rollout status deploy/inference --timeout=180s
	kubectl -n $(NS) rollout status deploy/api --timeout=180s

kind-e2e:
	kubectl -n $(NS) port-forward svc/api $(API_PORT):80 & PF=$$!; \
	sleep 3; tests/e2e.sh http://localhost:$(API_PORT); rc=$$?; kill $$PF; exit $$rc

kind-down:
	kind delete cluster --name $(CLUSTER)

bench: train
	PYTHONPATH=inference $(PY) benchmarks/model_bench.py --model inference/model/model.joblib
	$(PY) benchmarks/http_bench.py --url http://localhost:$(API_PORT) -c 8

# --- GPU (PyTorch) ---------------------------------------------------------
GPU_PY ?= .venv-gpu/bin/python

.PHONY: venv-gpu gpu-test gpu-bench gpu-image gpu-zip gpu-mock-demo

venv-gpu:
	python3 -m venv .venv-gpu && .venv-gpu/bin/pip install -r gpu/requirements.txt

gpu-test:
	$(GPU_PY) -m pytest gpu -q

gpu-bench:
	PYTHONPATH=inference $(GPU_PY) gpu/benchmark.py

gpu-image:
	docker build -f gpu/Dockerfile -t ai-inference-platform/gpu-bench:cpu .

# Upload bundle for Colab when the repo is not cloneable.
gpu-zip:
	mkdir -p dist && rm -f dist/gpu-bench.zip
	zip -q dist/gpu-bench.zip pyproject.toml gpu/textclf.py gpu/benchmark.py gpu/test_textclf.py \
		inference/app/__init__.py inference/app/data.py
	@echo "wrote dist/gpu-bench.zip"

# Scheduling demo on kind with a FAKE nvidia.com/gpu (see docs/gpu.md). The Job is expected to fail.
gpu-mock-demo: gpu-image
	kind load docker-image ai-inference-platform/gpu-bench:cpu --name $(CLUSTER)
	kubectl apply -k deployments/k8s/gpu-mock
	sleep 5; kubectl -n $(NS) get pods -l app=gpu-benchmark
	deployments/k8s/gpu/mock-gpu-node.sh add
	sleep 20; kubectl -n $(NS) get pods -l app=gpu-benchmark; kubectl -n $(NS) logs job/gpu-benchmark || true
	kubectl -n $(NS) delete job gpu-benchmark
	deployments/k8s/gpu/mock-gpu-node.sh remove
