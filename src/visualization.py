import numpy as np
import tensorflow as tf
import cv2

def generate_gradcam(model, image, class_idx):
    """
    Generate Grad-CAM visualization for the given image and class.
    """
    try:
        # Get the last convolutional layer
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break
        
        # Create a model that outputs both the last conv layer and the final output
        grad_model = tf.keras.Model(
            inputs=[model.inputs],
            outputs=[last_conv_layer.output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            class_score = predictions[:, class_idx]
        
        # Calculate gradients
        grads = tape.gradient(class_score, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels and create heatmap
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Resize and apply colormap
        heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[1]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        return heatmap
        
    except Exception as e:
        print(f"Error generating Grad-CAM: {str(e)}")
        return None

def overlay_heatmap(image, heatmap, alpha=0.4):
    """
    Overlay heatmap on original image.
    """
    image = (image * 255).astype(np.uint8)
    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlayed