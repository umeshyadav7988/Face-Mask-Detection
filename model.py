"""
Face Mask Detection - CNN Model Architecture
Uses MobileNetV2 transfer learning for lightweight, accurate mask classification.
"""

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    AveragePooling2D,
    Dense,
    Dropout,
    Flatten,
    Input,
)
from tensorflow.keras.models import Model


def build_mask_detector(input_shape=(224, 224, 3), num_classes=2):
    """
    Build a face mask detector model using MobileNetV2 as the base.

    Args:
        input_shape: Tuple of (height, width, channels) for input images.
        num_classes: Number of output classes (default: 2 — mask/no_mask).

    Returns:
        A compiled Keras Model ready for training.
    """
    # Load MobileNetV2 pre-trained on ImageNet, excluding the top classification head
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_tensor=Input(shape=input_shape),
    )

    # Freeze the base model layers — we only train our custom head
    for layer in base_model.layers:
        layer.trainable = False

    # Build custom classification head
    head = base_model.output
    head = AveragePooling2D(pool_size=(7, 7))(head)
    head = Flatten(name="flatten")(head)
    head = Dense(128, activation="relu")(head)
    head = Dropout(0.5)(head)
    head = Dense(num_classes, activation="softmax")(head)

    # Assemble the full model
    model = Model(inputs=base_model.input, outputs=head)

    return model


if __name__ == "__main__":
    model = build_mask_detector()
    model.summary()
    print("\n✅ Model built successfully!")
    print(f"   Total params: {model.count_params():,}")
