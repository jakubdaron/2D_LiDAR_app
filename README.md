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

The main window of the application allows the user to choose one of the 5 available application modes and to exit the application.

## Scan acquisition mode

** Single LiDAR data acquisition - Waveshare STL-27L **

https://github.com/user-attachments/assets/0af75419-1345-4458-8934-2fe530db2b8b

** Single LiDAR data acquisition - Slamtec A2M8 **

https://github.com/user-attachments/assets/7b4041e9-d50e-4f91-882e-2cfb5e5d4547

** Double LiDAR data acquisition - both devices **

https://github.com/user-attachments/assets/73cdd115-0e28-42ed-bfd5-9348b8b5e749

This mode allows live control of the scanning process for a single or both LiDAR devices, with the ability to adjust the data reading range and save scans to a CSV file.

## ICP algorithm mode

![icp](https://github.com/user-attachments/assets/ecb5975b-7fb9-4ed4-b36d-6582a773c94f)

This mode allows the alignment of two point clouds saved in a CSV file and displays the result on a chart.

## Scan Slicer mode

![slice](https://github.com/user-attachments/assets/2c7fe122-2dae-4672-b025-f119d44111ea)

This mode allows loading any scan in CSV format from the 'Scans' folder into the application window, followed by the ability to select a portion of the point cloud and save the selected fragment to a CSV file under a user-defined name.

## Calculate slice parameters mode

![analysis](https://github.com/user-attachments/assets/dc7fc73e-f595-4e92-adff-1fef692da3c2)

This mode allows loading any profile in CSV format from the 'Slices' folder into the application window, followed by the calculation of certain statistical parameters, such as RMS error, linear regression, Pearson coefficient, profile length, and the profile's distance from the LiDAR system.

## Calculate angle of measured profile mode

![degree](https://github.com/user-attachments/assets/e1e76044-25f3-45d0-ac88-a1e91c51a1b5)

This mode allows loading any L-shaped profile in CSV format from the 'Slices' folder into the application window, followed by calculating the angle between the walls of the L-shaped profile.
