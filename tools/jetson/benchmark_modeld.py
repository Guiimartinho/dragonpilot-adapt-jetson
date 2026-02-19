#!/usr/bin/env python3
"""
Benchmark modeld + dmonitoringmodeld on Jetson AGX Xavier.

Measures: inference latency, frame drops, GPU/CPU/memory usage, temperatures.
Run for N minutes and output a full report with comparison-ready CSV.

Usage:
  python3 -m tools.jetson.benchmark_modeld --duration 300  # 5 min
  python3 -m tools.jetson.benchmark_modeld --duration 3600 # 1 hour
"""
import os
import sys
import time
import argparse
import subprocess
import numpy as np

# Must be before cereal imports
os.environ.setdefault('DEV', 'CUDA')

import cereal.messaging as messaging


def get_gpu_usage():
  """Read Jetson GPU usage from sysfs."""
  try:
    with open('/sys/devices/gpu.0/load', 'r') as f:
      return int(f.read().strip()) / 10.0  # returns 0-1000, convert to %
  except Exception:
    return -1.0


def get_temperatures():
  """Read Jetson thermal zones."""
  temps = {}
  try:
    for i in range(10):
      type_path = f'/sys/class/thermal/thermal_zone{i}/type'
      temp_path = f'/sys/class/thermal/thermal_zone{i}/temp'
      if os.path.exists(type_path) and os.path.exists(temp_path):
        with open(type_path) as f:
          name = f.read().strip()
        with open(temp_path) as f:
          temp = int(f.read().strip()) / 1000.0
        temps[name] = temp
  except Exception:
    pass
  return temps


def get_cpu_usage():
  """Get CPU usage via /proc/stat snapshot."""
  try:
    with open('/proc/stat') as f:
      line = f.readline()
    parts = line.split()
    total = sum(int(x) for x in parts[1:])
    idle = int(parts[4])
    return total, idle
  except Exception:
    return 0, 0


def get_memory_usage():
  """Get memory usage in MB."""
  try:
    with open('/proc/meminfo') as f:
      lines = f.readlines()
    info = {}
    for line in lines:
      parts = line.split()
      info[parts[0].rstrip(':')] = int(parts[1])
    total = info.get('MemTotal', 0) / 1024
    available = info.get('MemAvailable', 0) / 1024
    return total, total - available
  except Exception:
    return 0, 0


def get_dla_usage():
  """Check if DLA engines are active."""
  try:
    result = subprocess.run(['cat', '/sys/kernel/debug/host1x/actmon_avg'], capture_output=True, text=True, timeout=1)
    return result.stdout.strip()
  except Exception:
    return "N/A"


def get_process_info(name):
  """Get CPU% and RSS for a process by name."""
  try:
    result = subprocess.run(
      ['pgrep', '-f', name],
      capture_output=True, text=True, timeout=2
    )
    pids = result.stdout.strip().split('\n')
    total_cpu = 0
    total_rss = 0
    for pid in pids:
      if not pid:
        continue
      try:
        with open(f'/proc/{pid}/stat') as f:
          stat = f.read().split()
        # RSS in pages
        rss = int(stat[23]) * 4096 / (1024 * 1024)  # MB
        total_rss += rss
      except Exception:
        pass
    return len([p for p in pids if p]), total_rss
  except Exception:
    return 0, 0


def main():
  parser = argparse.ArgumentParser(description='Benchmark modeld on Jetson')
  parser.add_argument('--duration', type=int, default=300, help='Duration in seconds (default: 300 = 5min)')
  parser.add_argument('--label', type=str, default='current', help='Label for this run (e.g. "tinygrad_only", "trt_dla")')
  parser.add_argument('--output', type=str, default='/tmp/benchmark_results.txt', help='Output file')
  args = parser.parse_args()

  print(f"=== Jetson modeld Benchmark ===")
  print(f"Label: {args.label}")
  print(f"Duration: {args.duration}s ({args.duration/60:.1f} min)")
  print(f"Output: {args.output}")
  print()

  sm = messaging.SubMaster(['modelV2', 'drivingModelData', 'driverStateV2', 'deviceState'])

  # Metrics storage
  model_times = []        # modeld inference time (ms)
  dmon_times = []          # dmonitoringmodeld inference time (ms)
  model_frame_ids = []     # frame IDs to detect drops
  gpu_usage_samples = []
  cpu_total_prev, cpu_idle_prev = get_cpu_usage()
  cpu_usage_samples = []
  mem_usage_samples = []
  temp_samples = []
  model_fps_samples = []

  start_time = time.monotonic()
  last_print = start_time
  last_cpu_check = start_time
  sample_count = 0
  model_count = 0
  dmon_count = 0
  model_count_window = 0
  fps_window_start = start_time

  print(f"{'Time':>6s} | {'Model ms':>9s} | {'DMon ms':>8s} | {'FPS':>5s} | {'GPU%':>5s} | {'CPU%':>5s} | {'RAM MB':>7s} | {'GPU°C':>6s} | {'CPU°C':>6s}")
  print("-" * 85)

  try:
    while True:
      elapsed = time.monotonic() - start_time
      if elapsed >= args.duration:
        break

      sm.update(100)

      if sm.updated['modelV2']:
        mv2 = sm['modelV2']
        t_ms = mv2.modelExecutionTime * 1000
        model_times.append(t_ms)
        model_frame_ids.append(mv2.frameId)
        model_count += 1
        model_count_window += 1

      if sm.updated['driverStateV2']:
        ds = sm['driverStateV2']
        if hasattr(ds, 'modelExecutionTime'):
          t_ms = ds.modelExecutionTime * 1000
          if t_ms > 0:
            dmon_times.append(t_ms)
            dmon_count += 1

      # Sample system metrics every 2 seconds
      now = time.monotonic()
      if now - last_cpu_check >= 2.0:
        gpu_pct = get_gpu_usage()
        gpu_usage_samples.append(gpu_pct)

        cpu_total, cpu_idle = get_cpu_usage()
        dt = cpu_total - cpu_total_prev
        di = cpu_idle - cpu_idle_prev
        cpu_pct = 100.0 * (1.0 - di / max(dt, 1))
        cpu_usage_samples.append(cpu_pct)
        cpu_total_prev, cpu_idle_prev = cpu_total, cpu_idle

        _, mem_used = get_memory_usage()
        mem_usage_samples.append(mem_used)

        temps = get_temperatures()
        temp_samples.append(temps)

        # FPS window
        window_elapsed = now - fps_window_start
        if window_elapsed > 0:
          fps = model_count_window / window_elapsed
          model_fps_samples.append(fps)
        model_count_window = 0
        fps_window_start = now

        last_cpu_check = now
        sample_count += 1

      # Print status every 10 seconds
      if now - last_print >= 10.0:
        gpu_pct = gpu_usage_samples[-1] if gpu_usage_samples else 0
        cpu_pct = cpu_usage_samples[-1] if cpu_usage_samples else 0
        mem_mb = mem_usage_samples[-1] if mem_usage_samples else 0
        fps = model_fps_samples[-1] if model_fps_samples else 0
        temps = temp_samples[-1] if temp_samples else {}
        gpu_temp = temps.get('GPU-therm', temps.get('gpu', 0))
        cpu_temp = temps.get('CPU-therm', temps.get('cpu', 0))
        model_med = np.median(model_times[-50:]) if model_times else 0
        dmon_med = np.median(dmon_times[-50:]) if dmon_times else 0

        mins = int(elapsed) // 60
        secs = int(elapsed) % 60
        print(f"{mins:3d}:{secs:02d} | {model_med:7.1f}ms | {dmon_med:6.1f}ms | {fps:5.1f} | {gpu_pct:5.1f} | {cpu_pct:5.1f} | {mem_mb:7.0f} | {gpu_temp:5.1f} | {cpu_temp:5.1f}")
        last_print = now

  except KeyboardInterrupt:
    print("\n[interrupted]")

  elapsed = time.monotonic() - start_time
  print(f"\n{'='*85}")
  print(f"=== RESULTS: {args.label} ({elapsed:.0f}s / {elapsed/60:.1f}min) ===")
  print(f"{'='*85}")

  # Modeld stats
  if model_times:
    mt = np.array(model_times)
    print(f"\n--- modeld inference ---")
    print(f"  Samples:  {len(mt)}")
    print(f"  Median:   {np.median(mt):.2f} ms")
    print(f"  Mean:     {np.mean(mt):.2f} ms")
    print(f"  Stddev:   {np.std(mt):.2f} ms")
    print(f"  P5:       {np.percentile(mt, 5):.2f} ms")
    print(f"  P95:      {np.percentile(mt, 95):.2f} ms")
    print(f"  P99:      {np.percentile(mt, 99):.2f} ms")
    print(f"  Min:      {np.min(mt):.2f} ms")
    print(f"  Max:      {np.max(mt):.2f} ms")
    print(f"  Avg FPS:  {len(mt)/elapsed:.1f}")

  # Frame drops
  if len(model_frame_ids) > 1:
    ids = np.array(model_frame_ids)
    diffs = np.diff(ids)
    drops = np.sum(diffs > 1)
    total_dropped = np.sum(diffs[diffs > 1] - 1)
    print(f"\n--- Frame drops ---")
    print(f"  Drop events: {drops}")
    print(f"  Total frames dropped: {total_dropped}")
    print(f"  Drop rate: {total_dropped / len(ids) * 100:.2f}%")

  # dmonitoringmodeld stats
  if dmon_times:
    dt = np.array(dmon_times)
    print(f"\n--- dmonitoringmodeld inference ---")
    print(f"  Samples:  {len(dt)}")
    print(f"  Median:   {np.median(dt):.2f} ms")
    print(f"  Mean:     {np.mean(dt):.2f} ms")
    print(f"  P95:      {np.percentile(dt, 95):.2f} ms")
  else:
    print(f"\n--- dmonitoringmodeld: no data (not running or no driver cam) ---")

  # GPU stats
  if gpu_usage_samples:
    gu = np.array(gpu_usage_samples)
    print(f"\n--- GPU usage ---")
    print(f"  Median:   {np.median(gu):.1f}%")
    print(f"  Mean:     {np.mean(gu):.1f}%")
    print(f"  Max:      {np.max(gu):.1f}%")

  # CPU stats
  if cpu_usage_samples:
    cu = np.array(cpu_usage_samples)
    print(f"\n--- CPU usage ---")
    print(f"  Median:   {np.median(cu):.1f}%")
    print(f"  Mean:     {np.mean(cu):.1f}%")
    print(f"  Max:      {np.max(cu):.1f}%")

  # Memory stats
  if mem_usage_samples:
    mu = np.array(mem_usage_samples)
    total_mem, _ = get_memory_usage()
    print(f"\n--- Memory ---")
    print(f"  Total:    {total_mem:.0f} MB")
    print(f"  Used avg: {np.mean(mu):.0f} MB")
    print(f"  Used max: {np.max(mu):.0f} MB")

  # Temperature stats
  if temp_samples:
    all_temps = {}
    for ts in temp_samples:
      for k, v in ts.items():
        if k not in all_temps:
          all_temps[k] = []
        all_temps[k].append(v)
    print(f"\n--- Temperatures ---")
    for k in sorted(all_temps.keys()):
      t = np.array(all_temps[k])
      print(f"  {k:20s}: avg={np.mean(t):.1f}°C  max={np.max(t):.1f}°C")

  # FPS stability
  if model_fps_samples:
    fps = np.array(model_fps_samples)
    print(f"\n--- FPS stability ---")
    print(f"  Median:   {np.median(fps):.1f} FPS")
    print(f"  Min:      {np.min(fps):.1f} FPS")
    print(f"  Max:      {np.max(fps):.1f} FPS")
    print(f"  Stddev:   {np.std(fps):.2f}")
    below_target = np.sum(fps < 14.0)
    print(f"  Below 14 FPS: {below_target}/{len(fps)} ({below_target/len(fps)*100:.1f}%)")

  # Save to file
  with open(args.output, 'w') as f:
    f.write(f"BENCHMARK: {args.label}\n")
    f.write(f"Duration: {elapsed:.0f}s\n")
    f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    if model_times:
      mt = np.array(model_times)
      f.write(f"modeld_median_ms={np.median(mt):.2f}\n")
      f.write(f"modeld_mean_ms={np.mean(mt):.2f}\n")
      f.write(f"modeld_p95_ms={np.percentile(mt, 95):.2f}\n")
      f.write(f"modeld_p99_ms={np.percentile(mt, 99):.2f}\n")
      f.write(f"modeld_max_ms={np.max(mt):.2f}\n")
      f.write(f"modeld_fps={len(mt)/elapsed:.1f}\n")

    if len(model_frame_ids) > 1:
      ids = np.array(model_frame_ids)
      diffs = np.diff(ids)
      total_dropped = np.sum(diffs[diffs > 1] - 1)
      f.write(f"frame_drop_rate={total_dropped / len(ids) * 100:.2f}%\n")

    if gpu_usage_samples:
      f.write(f"gpu_usage_median={np.median(gpu_usage_samples):.1f}%\n")
      f.write(f"gpu_usage_max={np.max(gpu_usage_samples):.1f}%\n")

    if cpu_usage_samples:
      f.write(f"cpu_usage_median={np.median(cpu_usage_samples):.1f}%\n")

    if mem_usage_samples:
      f.write(f"mem_used_avg_mb={np.mean(mem_usage_samples):.0f}\n")
      f.write(f"mem_used_max_mb={np.max(mem_usage_samples):.0f}\n")

    if temp_samples:
      all_temps_max = {}
      for ts in temp_samples:
        for k, v in ts.items():
          all_temps_max[k] = max(all_temps_max.get(k, 0), v)
      for k, v in sorted(all_temps_max.items()):
        f.write(f"temp_max_{k}={v:.1f}\n")

  print(f"\nResults saved to {args.output}")
  print(f"Run on the other version and compare!")


if __name__ == '__main__':
  main()
