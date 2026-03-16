import tensorflow as tf

print("Loading original model...")
# Load the model that works on your machine
model = tf.keras.models.load_model("models/dr_model_best.keras")

print("Converting and saving as H5...")
# Save it in the highly stable H5 format
model.save("models/dr_model_best.h5")

print("Success! You can now upload dr_model_best.h5 to GitHub.")