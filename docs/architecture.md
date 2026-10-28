# Architecture

```mermaid
flowchart LR
    client([Client]) -->|POST /v1/predict<br/>{"text": ...}| apiSvc

    subgraph cluster["Kubernetes namespace: ai-inference"]
        apiSvc[Service: api<br/>ClusterIP :80] --> apiPods
        subgraph apiPods["Deployment: api (Go) ×2"]
            api1[api pod]
            api2[api pod]
        end
        apiPods -->|POST /predict<br/>{"texts": [...]}| infSvc
        infSvc[Service: inference<br/>ClusterIP :8000] --> infPods
        subgraph infPods["Deployment: inference (Python) ×2"]
            inf1[inference pod<br/>TF-IDF + LogReg]
            inf2[inference pod<br/>TF-IDF + LogReg]
        end
        cm[(ConfigMap<br/>api-config)] -.->|INFERENCE_URL,<br/>INFERENCE_TIMEOUT| apiPods
    end
```

## Request flow

1. The client sends `POST /v1/predict` with `{"text": "..."}` to the **api** Service.
2. The Go API validates the body. It must be JSON with no unknown fields, at most 16 KiB, and the text must be non-blank UTF-8 of at most 2000 characters. Invalid input returns `400` without calling the inference service.
3. The API forwards `{"texts": [text]}` to `http://inference:8000/predict`, the cluster DNS name of the inference Service, with a 3 s timeout.
4. The inference service runs `predict_proba` and returns a label, its confidence and the score for every class.
5. The API maps the response to its public shape, or converts upstream failures into stable errors:

| Situation | Status |
|---|---|
| Invalid input | 400 |
| Inference returned non-200, malformed JSON or the wrong number of predictions | 502 |
| Inference unreachable (connection refused, DNS) | 502 |
| Inference slower than `INFERENCE_TIMEOUT` | 504 |

## Health model

| Endpoint | Probe | Meaning |
|---|---|---|
| api `/healthz` | liveness | The Go process is serving HTTP. It does **not** check dependencies, so an inference outage never restarts API pods. |
| api `/readyz` | readiness | The inference service's `/readyz` returns 200. While inference is down, API pods leave the Service endpoints. |
| inference `/healthz` | startup, liveness | The Python process is up. |
| inference `/readyz` | readiness | The model is loaded. A missing model leaves the pod alive but not ready, instead of crash-looping. |

## Model lifecycle

The model is trained **inside the image build** (`RUN python -m app.train`). Data generation and training use a fixed seed, so every build produces the same model. The `metrics.json` saved next to the model is exposed at `GET /metadata` on the inference service. [decisions.md](decisions.md) covers the trade-offs.
