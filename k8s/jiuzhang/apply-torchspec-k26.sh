#!/usr/bin/env bash
set -euo pipefail
cd /root/work/TorchSpec
kubectl create namespace torchspec-k26 --dry-run=client -o yaml | kubectl apply -f -
kubectl get secret novitalabs -n default -o json   | python3 -c 'import json, sys; o=json.load(sys.stdin); m=o["metadata"]; o["metadata"]={"name": m["name"], "namespace": "torchspec-k26"}; print(json.dumps(o))'   | kubectl apply -f -
kubectl apply -f k8s/jiuzhang/torchspec-k26-ray.yaml
