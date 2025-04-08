class PipelineOrchestrator:
    def __init__(self, pipelines):
        self.pipelines = pipelines

    def run(self):
        for pipeline in self.pipelines:
            pipeline.run()