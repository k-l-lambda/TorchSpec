#!/usr/bin/env bash
set -euo pipefail

check_local() {
  pid=
  if [ -n  ]; then
    ps -fp  2>/dev/null || true
  fi
  echo K2.6 files: 0
  du -sh /data1/models/Kimi-K2.6 /data1/models/lightseekorg-kimi-k2.5-eagle3 2>/dev/null || true
  tail -8 /data1/logs/torchspec_k26_download_*.log
}

echo '=== jiuzhang.51 ==='
check_local

echo '=== jiuzhang.65 ==='
ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new 172.31.13.65 'bash -s' <<'REMOTE'
pid=
if [ -n  ]; then
  ps -fp  2>/dev/null || true
fi
echo K2.6 files: 0
du -sh /data1/models/Kimi-K2.6 /data1/models/lightseekorg-kimi-k2.5-eagle3 2>/dev/null || true
tail -8 /data1/logs/torchspec_k26_download_*.log
REMOTE
