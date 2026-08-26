#!/usr/bin/env bash
# ============================================================
# run_campaign.sh — full S1-S5 validation campaign (thesis Ch.4)
# Produces results/summary.csv rows for Tables 4.2-4.4.
# Prereq: Gazebo world with the DD21 ALREADY running.
# Usage:  ./run_campaign.sh [N_RUNS]     (default 3 repetitions)
# ============================================================
set -e
N=${1:-3}
PKG=duckiedrone_validation

run () {  # controller scenario run_id mismatch
  echo "=== $1 / $2 / run $3 (mismatch=$4) ==="
  roslaunch $PKG run_scenario.launch controller:=$1 scenario:=$2 \
      run_id:=$3 mismatch:=$4 use_bridge:=true &
  LPID=$!
  # wait until scenario_runner exits (it breaks after landing)
  while kill -0 $LPID 2>/dev/null; do sleep 2; done
  sleep 3   # settle between runs
}

for CTRL in pid mpc vstmpc; do
  for SCN in S1 S2 S3 S4; do
    for R in $(seq 1 $N); do
      run $CTRL $SCN $R false
    done
  done
  for R in $(seq 1 $N); do
    run $CTRL S5 $R true        # S5 = trajectory under model mismatch
  done
done

echo "Campaign finished. See results/summary.csv and *_series.csv"
