from dataclasses import asdict
from pathlib import Path

import mlflow

from benchid.cost import usd_from_gpu_seconds
from benchid.jobs import GenJob, TrainSpec

EXPERIMENT = "piloto"


class Tracker:
    def __init__(self, out: Path, usd_per_hour: float) -> None:
        out = out.resolve()
        mlflow.set_tracking_uri(f"sqlite:///{(out / 'mlflow.db').as_posix()}")
        if mlflow.get_experiment_by_name(EXPERIMENT) is None:
            mlflow.create_experiment(EXPERIMENT, artifact_location=(out / "mlartifacts").as_uri())
        mlflow.set_experiment(EXPERIMENT)
        self.usd_per_hour = usd_per_hour

    def _run_id(self, key: str, kind: str, params: dict) -> str:
        found = mlflow.search_runs(filter_string=f"tags.clave = '{key}'", output_format="list")
        if found:
            return found[0].info.run_id
        with mlflow.start_run(run_name=key, tags={"clave": key, "tipo": kind}) as run:
            mlflow.log_params(params)
            return run.info.run_id

    def train(self, spec: TrainSpec, record: dict) -> None:
        with mlflow.start_run(run_id=self._run_id(spec.id, "entrenamiento", asdict(spec))):
            mlflow.set_tags({"fase": record["phase"], "backend": record["backend"]})
            for epoch, loss in enumerate(record["loss"], start=1):
                mlflow.log_metric("loss", loss, step=epoch)
            mlflow.log_metrics({
                "segundos": record["seconds"],
                "vram_gb": record["vram_gb"],
                "usd": usd_from_gpu_seconds(record["seconds"], self.usd_per_hour),
            })

    def image(self, job: GenJob, record: dict, image: Path) -> None:
        params = {"arm": job.arm, "train_id": job.train_id, "epoch": job.epoch, "weights": job.weights}
        with mlflow.start_run(run_id=self._run_id(job.config_key, "generacion", params)):
            mlflow.set_tags({"fase": record["phase"], "backend": record["backend"]})
            mlflow.log_artifact(str(image), artifact_path=f"{job.identity}/{job.scenario}")
            row = {**asdict(job), "seconds": record["seconds"], "vram_gb": record["vram_gb"], "file": image.name}
            mlflow.log_table({k: [v] for k, v in row.items()}, artifact_file="imagenes.json")
