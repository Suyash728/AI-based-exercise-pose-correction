import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import json

# ── GPU Memory Fix for 4GB VRAM ──────────────────────────
# This prevents TF from grabbing ALL your VRAM at once
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print("GPU memory growth enabled ✅")

# ── Settings — tuned for 4GB VRAM ───────────────────────
FRAMES_TRAIN = 'frames/train'
FRAMES_VAL   = 'frames/val'
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 16          # 32 will OOM on 4GB — keep at 16
EPOCHS       = 15

# ── Load Dataset ─────────────────────────────────────────
train_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,        # data augmentation
    rotation_range=10,           # slight rotation
    zoom_range=0.1               # slight zoom
)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    FRAMES_TRAIN,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
val_data = val_gen.flow_from_directory(
    FRAMES_VAL,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=list(train_data.class_indices.keys())  # only use classes from training set
)

NUM_CLASSES = train_data.num_classes
print(f"Training on {NUM_CLASSES} exercise classes")

# ── Build Model ───────────────────────────────────────────
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False  # freeze pretrained weights

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()  # prints model architecture

# ── Callbacks ─────────────────────────────────────────────
# Auto-save best model
checkpoint = ModelCheckpoint(
    'models/best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)
# Stop early if no improvement for 5 epochs
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

# ── Train ─────────────────────────────────────────────────
print("\nStarting training...")
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data,
    callbacks=[checkpoint, early_stop]
)

# ── Save Final Model & Labels ─────────────────────────────
model.save('models/exercise_classifier.h5')

index_to_class = {str(v): k for k, v in train_data.class_indices.items()}
with open('models/class_labels.json', 'w') as f:
    json.dump(index_to_class, f)

print("\n✅ Training complete! Model saved to models/")
print(f"Final validation accuracy: {max(history.history['val_accuracy']):.2%}")