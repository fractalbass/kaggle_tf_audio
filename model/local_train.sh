export MODEL="gs://kaggle_voice_data"
gcloud ml-engine local train --package-path trainer --module-name trainer.google_model_trainer --job-dir=$MODEL
