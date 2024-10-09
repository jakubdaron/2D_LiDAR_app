# 2D LiDAR app
2D LiDAR data acquisition and analysis app. The application works with Slamtec A2M8 and Waveshare STL-27L LiDARs. The application was created for the purposes of my thesis entitled "Research on the possibilities of using low-budget LIDAR devices in robotics".

# How to install the app?
To run the application, an environment that supports Python version 3.10 is required. Additionally, the installation of environment extensions such as the following is necessary:
- PyQt5 - 5.15.10
- matplotlib - 3.8.3
- numpy - 1.26.4
- scikit-learn - 1.4.1.post1
- scipy - 1.12.0
- pyserial - 3.3

# UI presentation and description
## Main window 

![main_w](https://github.com/user-attachments/assets/ac23c00e-161a-475f-b751-625891c2773b)

The main window of the application allows the user to choose one of the 5 available application modes and to exit the application

## Scan acquisition mode

** Single LiDAR data acquisition - Waveshare STL-27L **

https://github.com/user-attachments/assets/9d352926-f6a0-4818-944b-b5e4e5790a48

** Single LiDAR data acquisition - Slamtec A2M8 **

** Double LiDAR data acquisition - both devices **

## ICP algorithm mode

## Scan Slicer mode

## Calculate slice parameters mode

## Calculate angle of measured profile mode
