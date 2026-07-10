from comet_ml import start
from comet_ml.integration.pytorch import log_model
from ultralytics import YOLO

def main():
    experiment = start(
        api_key="hUCj6h5fuoZYtclQwhHDxIpWD",
        project_name="brain-tumor",
        workspace="pruthvi423186"
    )

    model = YOLO("yolo26n.pt")

    model.train(
        data="D:\\Datasets\\Data\\data.yaml",
        epochs=150,
        patience=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=4,
        amp=False,
        cache="disk",
        optimizer="auto",
        cos_lr=True,
        close_mosaic=10,
        pretrained=True,
        project="runs/train",
        name="yolo26n_custom",
        exist_ok=True,
        plots=True,
        seed=0,
    )

    log_model(experiment, model.model, model_name="yolo26n_brain_tumor")

    experiment.end()

if __name__ == "__main__":
    main()