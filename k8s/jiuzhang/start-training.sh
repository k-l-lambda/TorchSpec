#!/usr/bin/env bash
set -euo pipefail
kubectl exec -n torchspec-k26 torchspec-k26-head -- bash -lc "touch /data1/logs/torchspec-k8s/start-training"
