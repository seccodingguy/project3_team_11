from orchestrator import Orchestrator

orchestrator_config = {
'search_query': 'vinyl records and books',
'download_dir': 'data/train',
'validated_dir':'data/validated',
'train_dir':'data/train',
'validation_dir':'data/validation',
'model_dir':'models/vinyl_book_model.keras',
'test_dir':'data/test',
'prediction_dir':'data/predictions.json',
'model_name':'mobilenet',
'inventory_dir':'data/inventory',
'layers':15,
'agents':{'download_images':False,'validate_images':False,'train_model':True,'predict_images':True,'create_inventory':False},
'confidence' : 0.7,
'folds':12,
'epocs':100,
'use_saved_model':False,
'predict_dir':'test_images'
}

orchestrator_obj = Orchestrator(orchestrator_config)
orchestrator_obj.orchestrate_pipeline()
