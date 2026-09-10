class TrainPipelineConfig:
    def __init__(self):
        self.artifact_dir = "artifacts"


class TrainPipeline:
    def __init__(self):
        self.config = TrainPipelineConfig()

    def run_pipeline(self):
        print("Training pipeline started")