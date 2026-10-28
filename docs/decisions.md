# Design decisions

## 1. Separate Go API and Python inference service

**Decision:** The public API is a Go service. The model runs in its own Python service.

**Why:**
- The ML ecosystem (scikit-learn now, possibly PyTorch later) lives in Python, while request handling, validation and error mapping are simple and fast in Go.
- The two tiers scale independently. Inference is CPU- and memory-heavy, and would be GPU-bound later. The API is cheap.
- The model can be replaced without touching the public contract, because the API owns the response shape.

**Cost:** an extra network hop (≈3 ms p50 end to end at concurrency 1, compared with ≈0.6 ms for the model itself), plus two images to build and deploy.

## 2. TF-IDF + logistic regression, not a neural network

The goal is a correct, explainable, reproducible baseline. On this data the linear model reaches the noise ceiling (see below), so a bigger model would add latency and complexity for no measurable gain. A transformer is a reasonable next step only with real, harder data.

## 3. Synthetic data, explicitly labelled as such

There was no real support-ticket dataset with a licence suitable for a public repo. The generator in `inference/app/data.py` deliberately makes the task non-trivial:
- Generic openers and closers are shared by every class.
- 35% of tickets contain a phrase from another class ("also the courier is not answering").
- 5% of labels are flipped to simulate annotation noise.

**Implication for the metrics:** with 5% flipped labels, about 95% is roughly the best any model can score. The measured 95.4% accuracy means the model has learned the template signal, **not** that it would reach 95% on real tickets.

## 4. Split before fitting (no leakage)

`train_test_split(stratify=labels)` runs **before** the TF-IDF vectorizer is fitted. The vectorizer and classifier sit in one sklearn `Pipeline`, so the vocabulary and IDF weights come only from the training data. Test-set metrics therefore reflect unseen text.

## 5. Macro F1 as the headline metric

The classes are balanced here, so accuracy and macro F1 agree. Macro F1 is still reported because on real ticket data (where most tickets might be "technical") accuracy can look good while a small class like "account" is never predicted. Per-class precision and recall, and the confusion matrix, are in [metrics.json](metrics.json).

## 6. Train at image build time

**Pros:** the image is self-contained, startup needs no download, and builds are deterministic.
**Cons:** retraining requires a rebuild, and the image carries the training code.
**In production** the model would instead be a versioned artifact (an object store or model registry), loaded at startup and pinned by version in the Deployment.

## 7. Liveness does not check dependencies

If the API's liveness probe checked the inference service, an inference outage would make Kubernetes restart every healthy API pod, which only adds load and noise. Dependency health belongs in **readiness**, which stops traffic without restarting pods.

## 8. Security defaults

The images run as non-root numeric UIDs (`65532` for the API, `10001` for inference). The Pods set `runAsNonRoot`, drop all capabilities and block privilege escalation. The API also has a read-only root filesystem. No secrets are needed, so none are committed. Configuration lives in a ConfigMap.

## What I would change before production

- Run several uvicorn/gunicorn workers, or use an async-native server, for inference. The benchmark shows the Python HTTP layer, not the model, is the bottleneck.
- Version the model artifacts and load them from a registry instead of training in the build.
- Add request IDs propagated from API to inference, Prometheus metrics (latency histogram, error counts) and a HorizontalPodAutoscaler.
- Add a PodDisruptionBudget and anti-affinity so the replicas spread across nodes.
- Use an Ingress or Gateway with TLS instead of `port-forward`.
- Retrain and evaluate on real labelled tickets, monitor drift, and calibrate confidences before showing them to anyone.
