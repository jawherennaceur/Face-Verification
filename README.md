# Face Verification using Siamese Neural Network

This project implements a face verification system based on a Siamese Neural Network using TensorFlow and Keras. The model learns to distinguish between pairs of face images, predicting if they belong to the same person or not.

---

## Project Structure

- **Data preparation**  
  Loads and preprocesses images from three folders:  
  - `anchor/` : Reference face images  
  - `positive/` : Images of the same person (augmented by rotation and horizontal flip)  
  - `negative/` : Images of different people  

- **Model architecture**  
  A convolutional embedding network extracts 4096-dimensional feature vectors from input images.  
  A custom `Distance` layer calculates absolute difference between embeddings of image pairs.  
  A final Dense layer with sigmoid activation outputs similarity score.

- **Training**  
  Binary cross-entropy loss is used.  
  Adam optimizer with learning rate 1e-4.  
  Training runs for 5 epochs with batch size 16.  
  Precision, recall, and accuracy metrics are tracked during training.

- **Evaluation**  
  Model is tested on a held-out dataset and predictions are visualized.  
  Model weights are saved to `face_verif.h5`.

---

## Usage

1. **Prepare dataset**  
   Organize face images in three directories named `anchor`, `positive`, and `negative`.  
   Images are resized to 105x105 pixels.

2. **Run training**  
   Execute the training loop on the prepared dataset.

3. **Evaluate model**  
   Use test batches to generate predictions and visualize sample image pairs.

4. **Save/Load model**  
   Save model after training using `model.save('face_verif.h5')`.  
   Load model with custom `Distance` layer:
   ```python
   model = tf.keras.models.load_model('face_verif.h5', custom_objects={'Distance':Distance})
   ```

5. **Test the model**  
   Use `test_data` to predict similarity between new pairs:
   ```python
   test_input, test_val, y_true = test_data.as_numpy_iterator().next()
   y_pred = siamese_model.predict([test_input, test_val])
   predictions = [1 if p > 0.5 else 0 for p in y_pred]
   ```

6. **Visualize results**  
   Plot any test pair for inspection:
   ```python
   import matplotlib.pyplot as plt

   plt.figure(figsize=(10,4))
   plt.subplot(1,2,1)
   plt.title("Anchor")
   plt.imshow(test_input[0])

   plt.subplot(1,2,2)
   plt.title("Test Image")
   plt.imshow(test_val[0])
   plt.show()
   ```

---

## File Summary

- `face_verif.h5` : trained model file  
- `Distance` : custom layer used for comparing embeddings  
- Training and evaluation scripts are provided in a single notebook or `.py` script  
- Images: grouped in `anchor`, `positive`, and `negative` folders under the project directory

---

## Metrics

After 5 epochs:

- Accuracy: depends on threshold used in evaluation  
- Recall: ~0.98  
- Precision: 1.0  
- Loss: decreases from ~0.69 to near zero

These scores are based on binary predictions using a 0.5 threshold. You can fine-tune this threshold for better precision-recall tradeoff.

---

## Improvements

- Use larger and more diverse datasets  
- Replace manual loss loop with `model.fit` for faster training  
- Try contrastive loss or triplet loss instead of binary cross-entropy  
- Use pretrained embeddings (e.g., FaceNet, VGGFace2)  
- Add image augmentation (brightness, zoom, noise)

---

## Requirements

- Python 3.8+  
- TensorFlow 2.x  
- NumPy  
- OpenCV  
- scikit-image  
- Pillow  
- Matplotlib

Install dependencies with:
```bash
pip install tensorflow numpy opencv-python scikit-image pillow matplotlib
```
