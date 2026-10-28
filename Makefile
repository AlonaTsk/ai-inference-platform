CLUSTER  ?= ai-inference
NS       ?= ai-inference
API_PORT ?= 8080
PY       ?= .venv/bin/python

.PHONY: venv test test-go test-py train build up down e2e kind-up kind-deploy kind-e2e kind-down bench

venv:
	python3 -m venv .venv && .venv/bin/pip install -r inference/requirements-dev.txt

test: test-go test-py

test-go:
	cd api && go vet ./... && go test -race ./...

test-py:
	cd inference && ../$(PY) -m pytest -q

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
	$(PY) benchmarks/model_bench.py --model inference/model/model.joblib
	$(PY) benchmarks/http_bench.py --url http://localhost:$(API_PORT) -c 8
