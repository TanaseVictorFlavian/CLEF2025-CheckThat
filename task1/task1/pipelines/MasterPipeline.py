import json

class MasterPipeline:
    def __init__(self, config,  ingestion_pipeline, training_pipeline, evaluation_pipeline):
        self.ingestion_pipeline = ingestion_pipeline
        self.training_pipeline = training_pipeline
        self.evaluation_pipeline = evaluation_pipeline
        
    
    def run(self, save_run_info: bool = True):
        self.ingestion_pipeline.run()
        self.training_pipeline.run()
        self.evaluation_pipeline.run()
        
        run_info =self.log_run()
        
        if save_run_info:
            with open(self.config.run_info_path, "w") as f:
                json.dump(run_info, f)
            
    def log_run(self):
        run_info = {}
        run_info["language"] = self.ingestion_pipeline.language
        run_info["hyperparams"] = self.training_pipeline.hyperparams
        run_info["model_arch"] = self.training_pipeline.model.get_architecture()
        run_info["stats"] = self.evaluation_pipeline.get_stats()

        
        return run_info
        
        
    
        