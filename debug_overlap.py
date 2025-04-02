import os

training_images = []
for folder in os.listdir("../CO3D_ClickMe2"):
    for image in os.listdir("../CO3D_ClickMe2/"+folder):
        name = f"{folder}_{image}"
        name = name.replace('/', '_')
        training_images.append(f"{folder}_{image}")

# validation_images = []
# for folder in os.listdir("../CO3D_ClickMe_Training2"):
#     for image in os.listdir("../CO3D_ClickMe_Training2/"+folder):
#         validation_images.append(f"{folder}_{image}")

validation_images = []
for image in os.listdir("../human_clickme_data_processing/assets/co3d_train/"):
    name = image.replace('.npy', '')
    validation_images.append(f"{name}")

print(training_images[0], validation_images[0])

training_images = set(training_images)
validation_images = set(validation_images)

print(f"Lengths: Training={len(training_images)} Validation={len(validation_images)}")

print(len(training_images & validation_images))