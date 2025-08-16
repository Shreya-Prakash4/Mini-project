# Automated Crack Detection System using CNN

## 1. Introduction 
Infrastructure monitoring is crucial for public safety, but manual inspection methods are not scalable and often inaccurate.
This project focuses on detecting cracks in infrastructure images using deep learning techniques.I t automates the traditionally manual, time-consuming, and error-prone process of crack inspection.

- **Problem it solves**: Provides real-time and reliable crack detection, reducing human error.  
- **Usefulness**: Helps in infrastructure safety, maintenance planning, and preventing accidents due to unnoticed structural damage.
- **Necessity**: Ensures faster, reliable, and real-time crack monitoring.  
- **Technologies Used**:  
  - Python  
  - TensorFlow/Keras  
  - OpenCV (for image preprocessing)  
  - Matplotlib/Seaborn (for visualization)

## 2. Features
- 📸 **Automatic Crack Detection** in images.  
- 📊 **High Accuracy** (97% achieved using custom CNN).  
- 🖼️ Support for large-scale image datasets.  
- 🎥 Extendable for real-time crack detection in videos/CCTV footage.  
- 🔎 Dataset balanced with **positive & negative samples** (15k each).

## 3. Dataset / Resources
We used a **balanced dataset** for training and evaluation.  

- 📂 **Size**: 30,000 images (15,000 cracked + 15,000 non-cracked).  
- 🏷️ **Format**: `.jpg` and `.png` images.  
- 🌐 **Dataset Link**: [Concrete Crack Images Dataset (Kaggle)](https://www.kaggle.com/datasets/sherlockholmes/concrete-crack-images)

## 4. Project Structure
```plaintext
mini project/
|--miniproject(new)         #code used to preprocess and train model
|--data/                    #dataset further split into train(70%), test(15%) and, validate(15%)
|    |--train/
|    |--test/
|    |--validation/
|--project_folder/          #frontend and backend
     |--static/
         |--images/
             |--background2.png
         |--style.css
     |--templates/
         |--index.html
     |--app.py                #flask file
     |--my_cnn_model.keras    #trained model
```

## 5. Run the code
To run the code simply go to the directory where app.py file is present and run python app.py command and visit the website where it is hosted.

## 8. Results
- Accuracy: 97%
- Works well on unseen data
- Detects minute cracks effectively.

  
![02](https://github.com/user-attachments/assets/88e32e22-50b5-4c42-8a3a-57b8bdf40554)
![03](https://github.com/user-attachments/assets/a5458a7b-9fdf-4b53-8f0f-d9b3cfae0222)
![1100](https://github.com/user-attachments/assets/f9c35266-6e18-420b-bfe5-00cc15bee91f)
![02148](https://github.com/user-attachments/assets/e731f20c-b882-4fa9-bf33-56481ddee701)

## 9. Conclusion / Future Scope
- ✅ Successfully built a deep learning-based crack detection system with high accuracy.
- 🌍 Can be deployed in real-time monitoring systems (CCTV, drones).
- 🔮**Future improvements:**
    - Deployment with IoT devices for live monitoring.
    - Extending to classify crack severity (minor, moderate, severe).
    - Integrating with mobile/edge devices for field inspections.







