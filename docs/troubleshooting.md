# Troubleshooting log and diagnostic commands

The incidents below happened while building this project on a local kind cluster. The output is copied from the terminal, with pod-name hashes trimmed.

## Incident 1: API pods stuck in `CreateContainerConfigError`

**Symptom** after the first `kubectl apply -k deployments/k8s`:

```text
$ kubectl -n ai-inference get pods
NAME                        READY   STATUS                       RESTARTS   AGE
api-589fc4cb5c-p7cs7        0/1     CreateContainerConfigError   0          30s
api-589fc4cb5c-vmt7b        0/1     CreateContainerConfigError   0          30s
inference-d7b6fccff-fvft8   1/1     Running                      0          30s
inference-d7b6fccff-h5qmk   1/1     Running                      0          30s
```

**Diagnosis:** `kubectl describe` shows the reason in the Pod's events:

```text
$ kubectl -n ai-inference describe pod -l app=api
Events:
  Warning  Failed  8s (x4 over 37s)  kubelet  Error: container has runAsNonRoot and image has
  non-numeric user (nonroot), cannot verify user is non-root
```

**Root cause:** The Deployment sets `securityContext.runAsNonRoot: true`. The API Dockerfile had `USER nonroot:nonroot`. The kubelet cannot resolve a *user name* inside the image, so it cannot prove the user is not UID 0, and it refuses to start the container. The inference image worked because it uses a numeric UID (`USER 10001`).

**Fix** (`api/Dockerfile`): use the numeric UID that distroless's `nonroot` user maps to:

```dockerfile
USER 65532:65532
```

Then rebuild, reload the image into kind and roll out:

```text
$ docker compose build api
$ kind load docker-image ai-inference-platform/api:dev --name ai-inference
$ kubectl -n ai-inference rollout restart deploy/api
$ kubectl -n ai-inference rollout status deploy/api
deployment "api" successfully rolled out
```

**Lesson:** `CreateContainerConfigError` means the container spec could not be turned into a runnable container. The image pulled fine, so the cause is configuration or security context, and the Events section of `kubectl describe` names it.

## Incident 2 (deliberate): inference outage and readiness

To verify the readiness design, the inference Deployment was scaled to zero:

```text
$ kubectl -n ai-inference scale deploy/inference --replicas=0
$ kubectl -n ai-inference get pods
NAME                  READY   STATUS    RESTARTS   AGE
api-d88c7b475-dnm48   0/1     Running   0          34s
api-d88c7b475-jnkk6   0/1     Running   0          33s

$ kubectl -n ai-inference get endpoints api
NAME   ENDPOINTS   AGE
api                91s
```

- The API pods stayed **Running** (liveness passed, so there were no restarts) but became **0/1 Ready**, and the `api` Service had **no endpoints**. That is the intended behaviour.
- A request sent through `kubectl port-forward svc/api` still reached a pod and returned `502 Bad Gateway`. `port-forward` connects directly to one pod and **bypasses Service endpoint selection**, so readiness does not apply to it. Real in-cluster or ingress traffic would have been stopped.
- The API logged the root cause:

```json
{"level":"ERROR","msg":"inference call failed","err":"Post \"http://inference:8000/predict\": dial tcp 10.96.145.223:8000: connect: connection refused","status":502}
```

Scaling back to 2 replicas made both API pods Ready again within one readiness period, without any restarts.

## Incident 3: host port already allocated (Docker Compose)

```text
Bind for 0.0.0.0:8080 failed: port is already allocated
```

Another local project already published port 8080. The Compose file now reads the host port from `API_PORT` (default 8080): `API_PORT=18081 docker compose up -d`.

## Incident 4: latency regression from a sync FastAPI dependency

**Symptom:** after a refactor that moved the loaded model from a module-level dict into `app.state`, read through a FastAPI dependency (`Depends(loaded_model)`), the end-to-end benchmark looked slower. The laptop was heavily loaded at the time (load average about 11), so a before/after comparison of single runs was meaningless.

**Measuring it properly:**
- The previous commit was checked out with `git worktree`, built and run as a second Compose project next to the new one.
- Each inference service was benchmarked **directly from inside its own container**, removing the Go API and Docker networking from the measurement.
- 6 alternating rounds of 1000 requests each, so both versions saw the same machine load.

| version | p50 per round (ms) | median |
|---|---|---:|
| before refactor | 3.76 3.79 4.07 4.31 4.59 4.63 | 4.19 |
| after refactor | 4.05 4.46 4.71 4.91 5.01 5.44 | 4.81 |

**Root cause:** the dependency was declared with plain `def`. FastAPI runs `def` dependencies in its thread pool, the same as `def` endpoints. Every request therefore paid for an extra hand-off to a worker thread and back, just to read an attribute.

**Fix:** `async def loaded_model(...)`. It does no I/O, so it is safe to run on the event loop. After the fix, both versions measured 3.53–3.66 ms per round, apart from one 4.49 ms outlier in the old version: a regression of ≈0.

**Lessons:**
- In FastAPI, `def` versus `async def` is a scheduling decision. Use `def` for blocking work, such as the CPU-bound `predict` endpoint. Use `async def` for cheap non-blocking code.
- On a noisy machine, compare versions side by side with interleaved runs, never against a number recorded earlier.

The same comparison showed that raising the Go client's idle-connection pool (the default is 2 per host) improved API throughput by only about 5% at concurrency 8–32, within noise. The single Python worker remains the bottleneck.

## Diagnostic cheat sheet

```bash
kubectl -n ai-inference get pods,svc,endpointslices -o wide   # what is running, what is routable
kubectl -n ai-inference describe pod <pod>                     # Events: scheduling, pulls, probe failures
kubectl -n ai-inference logs deploy/api --tail=50              # JSON request/error logs
kubectl -n ai-inference logs <pod> --previous                  # logs from the crashed container
kubectl -n ai-inference rollout status deploy/api              # wait for and verify a rollout
kubectl -n ai-inference rollout undo deploy/api                # roll back a bad release
kubectl -n ai-inference get events --sort-by=.lastTimestamp    # namespace-wide timeline
kubectl -n ai-inference port-forward svc/api 8080:80           # local access (bypasses readiness)
kubectl -n ai-inference exec deploy/inference -- python -c \
  "import urllib.request;print(urllib.request.urlopen('http://localhost:8000/readyz').read())"
```
