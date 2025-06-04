# Models Directory

This directory contains trained models and related files for the AeraSync Feed project.

## Structure

```
models/
├── yamnet_finetuned/      # Fine-tuned YamNet model
│   ├── final_model.h5     # Final trained model
│   ├── best_model.h5      # Best model checkpoint
│   ├── label_encoder.json # Label encoder mapping
│   ├── training_results.json # Training metrics
│   ├── training_history.png  # Training plots
│   └── confusion_matrix.png  # Confusion matrix
├── yamnet.tflite          # TensorFlow Lite model for deployment
├── tflite_test_results.json # TFLite model test results
└── README.md              # This file
```

## Model Files

- **final_model.h5**: Complete trained TensorFlow/Keras model
- **best_model.h5**: Best model checkpoint during training
- **yamnet.tflite**: Optimized TensorFlow Lite model for Raspberry Pi
- **label_encoder.json**: Class names and label mappings

## Deployment

For Raspberry Pi deployment, use the `.tflite` model file:

```bash
python scripts/infer.py --model models/yamnet.tflite --labels models/yamnet_finetuned/label_encoder.json
```

## Model Performance

Check `training_results.json` for:

- Test accuracy
- Classification report
- Training parameters
- Model metrics
