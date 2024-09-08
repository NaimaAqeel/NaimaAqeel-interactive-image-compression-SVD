import numpy as np
import gradio as gr
from skimage import io
from skimage.transform import resize
from PIL import Image

def svd_compress(image_channel, k):
    """Compress a single channel of the image using SVD by keeping only the top k singular values."""
    U, S, Vt = np.linalg.svd(image_channel, full_matrices=False)
    compressed_channel = np.dot(U[:, :k], np.dot(np.diag(S[:k]), Vt[:k, :]))
    return compressed_channel

def resize_image(image_np, target_shape=(500, 500)):
    """Resize the image to reduce the computation time for SVD."""
    return resize(image_np, target_shape, anti_aliasing=True)

def process_image(image, k):
    """Process the uploaded image, compress it using SVD for each color channel, and return the result."""
    # Convert PIL Image to NumPy array
    image_np = np.array(image)
    
    # Resize the image to speed up SVD computation
    image_np_resized = resize_image(image_np)
    
    # Separate the RGB channels
    if len(image_np_resized.shape) == 3:  # Color image
        r_channel, g_channel, b_channel = image_np_resized[:, :, 0], image_np_resized[:, :, 1], image_np_resized[:, :, 2]
        
        # Compress each channel using SVD
        r_compressed = svd_compress(r_channel, k)
        g_compressed = svd_compress(g_channel, k)
        b_compressed = svd_compress(b_channel, k)
        
        # Stack the compressed channels back together
        compressed_image = np.stack([r_compressed, g_compressed, b_compressed], axis=2)
    else:  # Grayscale image
        compressed_image = svd_compress(image_np_resized, k)
    
    # Clip the values to ensure valid pixel range and convert to PIL Image for output
    compressed_image = np.clip(compressed_image * 255, 0, 255)
    compressed_image_pil = Image.fromarray(compressed_image.astype(np.uint8))
    
    return compressed_image_pil

# Gradio interface
gr.Interface(fn=process_image,
             inputs=[gr.Image(type="pil", label="Upload Image"),
                     gr.Slider(1, 100, step=1, value=50, label="Compression Rank")],
             outputs=gr.Image(type="pil", label="Compressed Image"),
             title="Color Image Compression using SVD",
             description="Upload an image (color or grayscale), and adjust the compression rank.").launch()
