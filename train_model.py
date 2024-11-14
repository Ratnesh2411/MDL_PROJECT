# Import TensorFlow library
import tensorflow as tf

# Import specific modules for building the model and layers
# Import the Sequential class to create a linear stack of neural network layers
from tensorflow.keras.models import Sequential

# Import layers for building a Convolutional Neural Network (CNN)
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Import the CIFAR-10 dataset which consists of 60,000 32x32 color images in 10 classes
from tensorflow.keras.datasets import cifar10


# Load the CIFAR-10 dataset (which consists of 60,000 32x32 color images in 10 classes)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the pixel values of the images to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define a function to create a Convolutional Neural Network (CNN) model
def create_model():
    # Create a Sequential model (layers will be added one after the other)
    model = Sequential([
        # Add a Conv2D layer with 32 filters, 3x3 kernel size, and ReLU activation
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        
        # Add a MaxPooling2D layer with a 2x2 pool size to downsample the feature maps
        MaxPooling2D((2, 2)),
        
        # Add another Conv2D layer with 64 filters, 3x3 kernel size, and ReLU activation
        Conv2D(64, (3, 3), activation='relu'),
        
        # Add another MaxPooling2D layer to further downsample the feature maps
        MaxPooling2D((2, 2)),
        
        # Add another Conv2D layer with 64 filters, 3x3 kernel size, and ReLU activation
        Conv2D(64, (3, 3), activation='relu'),
        
        # Flatten the 3D feature maps into a 1D vector
        Flatten(),
        
        # Add a Dense (fully connected) layer with 64 neurons and ReLU activation
        Dense(64, activation='relu'),
        
        # Add the output layer with 10 neurons (one for each class) and softmax activation
        Dense(10, activation='softmax')
    ])
    # Return the constructed model
    return model

# Create the CNN model using the defined function
model = create_model()

# Compile the model with Adam optimizer, sparse categorical crossentropy loss, and accuracy as a metric
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model on the training data for 10 epochs, using test data for validation
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Save the trained model to the specified path for future use
model.save('classifier/cifar10_model.h5')

# Print a confirmation message that the model has been trained and saved
print("Model training complete and saved as cifar10_model.h5.")
