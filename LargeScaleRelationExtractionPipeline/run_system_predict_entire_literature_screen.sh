#!/bin/bash

DIR=$(pwd)

JOBDIR="${DIR}/jobs/"
MAX_JOBS=76
TOTAL_FOLDERS=700

for ((PARAM_INPUT_FOLDER=1; PARAM_INPUT_FOLDER<=TOTAL_FOLDERS; PARAM_INPUT_FOLDER++)); do

  while true; do
      jobs=$(ls "$JOBDIR" | wc -l)
      if [ $jobs -lt $MAX_JOBS ]; then break; fi
      echo "Too many jobs, sleeping ..."
      sleep 60
  done

  echo "Submitting job with params" $PARAM_INPUT_FOLDER
  job_id=$(
      sbatch "${DIR}/puhti_single.sh" \
              $PARAM_INPUT_FOLDER \
              | perl -pe 's/Submitted batch job //'
  )
  echo "Submitted batch job $job_id"
  touch "$JOBDIR"/$job_id
  sleep 10
done
