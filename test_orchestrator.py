from orchestrator import Orchestrator

orchestrator_config = {
'search_query': 'vinyl records and books',
'download_dir': 'data/train',
'train_dir':'data/train',
'model_dir':'models/vinyl_book_model.keras',
'test_dir':'data/test',
'prediction_dir':'data/predictions.json',
'model_name':'mobilenet',
'layers':15,
'download_images':False,
'validate_images':False,
'train_model':True,
'predict_images':True,
'confidence' : 0.7,
'folds':12,
'epocs':100,
'use_saved_model':False,
'predict_dir':'test_images',
'use_xla':False
}

orchestrator_obj = Orchestrator(orchestrator_config)
orchestrator_obj.orchestrate_pipeline()
