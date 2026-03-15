import numpy as np
import tensorflow as tf
import cv2

def get_gradcam_heatmap(img_array, model):
    """
    Extracts the gradients from the final convolutional layer of the EfficientNet model
    to determine which pixels influenced the final prediction the most.
    """
    # 1. Extract the pre-trained EfficientNet base model
    base_model = model.layers[0]
    
    # 2. Find the last convolutional layer dynamically
    last_conv_layer_name = None
    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break
            
    # 3. Create a sub-model that outputs the last conv layer's activations
    conv_model = tf.keras.Model(base_model.inputs, base_model.get_layer(last_conv_layer_name).output)
    
    # 4. Create a sub-model that takes those activations and outputs the final prediction
    classifier_input = tf.keras.Input(shape=conv_model.output.shape[1:])
    x = classifier_input
    for layer in model.layers[1:]:
        x = layer(x)
    classifier_model = tf.keras.Model(classifier_input, x)
    
    # 5. Compute the gradients using TensorFlow's GradientTape
    with tf.GradientTape() as tape:
        last_conv_layer_output = conv_model(img_array)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]
        
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    
    # Weight the feature map by the computed gradients
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
        
    # Create the 2D heatmap
    heatmap = np.mean(last_conv_layer_output, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    
    return heatmap

def overlay_heatmap(img_bytes, heatmap, alpha=0.6):
    """
    Superimposes the generated heatmap onto the original uploaded image.
    """
    # Load original image from the raw bytes uploaded by the user
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize the heatmap to match the original high-resolution image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap into a colored "Jet" map (Red = high importance, Blue = low)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Blend the two images together
    superimposed_img = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)
    return superimposed_img