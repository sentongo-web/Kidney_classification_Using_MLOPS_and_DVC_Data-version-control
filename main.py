from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_model_trainer import ModelTrainerTrainingPipeline
from cnnClassifier.pipeline.stage_04_model_evaluation import ModelEvaluationPipeline

STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    DataIngestionTrainingPipeline().main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<" + chr(10) + chr(10) + "x==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Prepare Base Model stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    PrepareBaseModelTrainingPipeline().main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<" + chr(10) + chr(10) + "x==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Training stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    ModelTrainerTrainingPipeline().main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<" + chr(10) + chr(10) + "x==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Evaluation stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    ModelEvaluationPipeline().main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<" + chr(10) + chr(10) + "x==========x")
except Exception as e:
    logger.exception(e)
    raise e
