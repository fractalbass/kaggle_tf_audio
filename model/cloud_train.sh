#!/bin/bash
if [$1 eq ""]
then
    echo Please include a job name!
else
    rm -rf *.pyc
    rm -rf ./train/*.pyc
    gcloud ml-engine jobs submit training $1 --module-name trainer.google_model_trainer --package-path=trainer --job-dir=gs://kaggle_voice_data
    echo Training started.
fi

