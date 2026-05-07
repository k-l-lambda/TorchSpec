# Jiuzhang TorchSpec K2.6 Kubernetes setup

- Head/training node: `host-172-31-13-51` (`jiuzhang.51`)
- Worker/inference node: `host-172-31-13-65` (`jiuzhang.65`)
- Data/model root: `/data1`
- TorchSpec source: `/root/work/TorchSpec`
- K2.6 model: `/data1/models/Kimi-K2.6`
- Eagle3 draft model: `/data1/models/lightseekorg-kimi-k2.5-eagle3`
- Smoke config: `configs/sglang_kimi_k26_2node_jiuzhang_smoke.yaml`

Downloads use the Jiuzhang proxy: `http://127.0.0.1:1081`.

Use:

```bash
k8s/jiuzhang/check-downloads.sh
k8s/jiuzhang/apply-torchspec-k26.sh
kubectl get pods -n torchspec-k26 -o wide
k8s/jiuzhang/start-training.sh
kubectl logs -n torchspec-k26 torchspec-k26-head -f
```

The smoke config uses TCP Mooncake first. Switch to RDMA only after pod-visible IB devices are validated.
