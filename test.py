from deepface import DeepFace

# Verify two images
result = DeepFace.verify("./pics/img1.jpeg", "./pics/img2.jpeg")
print("Is verified: ", result["verified"])