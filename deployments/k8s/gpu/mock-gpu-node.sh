#!/usr/bin/env bash
# Advertise (or remove) a FAKE nvidia.com/gpu extended resource on a kind node.
#
# This only changes what the scheduler believes the node has. No GPU, driver or
# device is added, so a pod scheduled here has no CUDA. Use it to test
# scheduling behaviour (Pending -> Scheduled), never to produce benchmark numbers.
# See: https://kubernetes.io/docs/tasks/administer-cluster/extended-resource-node/
#
# Usage: deployments/k8s/gpu/mock-gpu-node.sh add|remove [node]
set -euo pipefail

action="${1:?usage: $0 add|remove [node]}"
node="${2:-ai-inference-control-plane}"

case "$action" in
  add)    patch='[{"op":"add","path":"/status/capacity/nvidia.com~1gpu","value":"1"}]' ;;
  remove) patch='[{"op":"remove","path":"/status/capacity/nvidia.com~1gpu"}]' ;;
  *)      echo "unknown action: $action" >&2; exit 1 ;;
esac

kubectl patch node "$node" --subresource=status --type=json -p "$patch"
kubectl get node "$node" -o jsonpath='{"nvidia.com/gpu allocatable: "}{.status.allocatable.nvidia\.com/gpu}{"\n"}'
