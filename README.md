# Hotel Booking Prediction

This project uses deep learning to predict whether a user will book a hotel based on their search behavior. The model analyzes various features such as user location, search details, hotel attributes, and booking preferences to improve prediction accuracy.

## Dataset

### Dataset Description

The dataset contains user search logs and hotel booking data collected from an online travel platform. It includes user information, search details, and hotel characteristics. Each entry in the dataset contains the following columns:

**Note:** The dataset is proprietary, and thus, its publication is not permitted.

| Column | Description |
|---|---|
| user | User ID |
| user_location_country | User's country |
| user_location_region | User's region |
| user_location_city | User's city |
| destination_distance | Distance between user and hotel |
| search_date | Date of search |
| is_mobile | Whether the search was made on a mobile device |
| is_package | Whether the search included a hotel and transportation package |
| channel | Channel through which the user came to the site |
| search_count | Number of similar searches at the time of the user's search |
| checkIn_date | Check-in date |
| checkOut_date | Check-out date |
| n_adults | Number of adults |
| n_children | Number of children |
| n_rooms | Number of rooms |
| destination | Destination ID |
| destination_type | Destination type code |
| hotel_continent | Hotel continent ID |
| hotel_country | Hotel country ID |
| hotel_market | Hotel market |
| hotel_category | Hotel category |
| is_booking | Whether the hotel was booked |


## Feature Engineering and Data Preprocessing

The following steps were taken to preprocess the data:

- **Handling Missing Values**:  
  - Missing values in the `destination_distance` column were filled with `0`, indicating an unknown distance.  
  - Rows with missing `checkIn_date` or `checkOut_date` were dropped.  

- **Date Feature Extraction**:  
  - The `search_date`, `checkIn_date`, and `checkOut_date` columns were converted to datetime format.  
  - Features such as `duration` (length of stay) and `days_between` (days between search and check-in) were derived.  

- **Categorical Encoding**:  
  - One-hot encoding was applied to categorical columns like `channel`.  

- **Seasonal Feature**:  
  - A `season` feature was created by mapping months to seasons (Winter, Spring, Summer, Fall).  

- **Abroad Feature**:  
  - A binary feature `is_abroad` was created to indicate whether the user's country differs from the hotel's country.  


## Model Training

The neural network is structured as follows:

1. **Input & Initial Normalization:**  
   - **Input Layer:** Receives the preprocessed feature vector.  
   - **Batch Normalization:** Normalizes the input data to stabilize and accelerate training.

2. **First Hidden Block:**  
   - **Dense Layer (100 units, ReLU):** Processes the normalized input.  
   - **Dropout (10%):** Applies regularization to prevent overfitting.  
   - **Batch Normalization:** Normalizes the output from the dense layer.

3. **Second Hidden Block:**  
   - **Dense Layer (50 units, ReLU):** Further extracts features from the data.  
   - **Batch Normalization:** Normalizes the activations to improve convergence.

4. **Output Layer:**  
   - **Dense Layer (1 unit, Sigmoid):** Produces the final booking probability.


The model was compiled using the Adam optimizer, binary cross-entropy loss, and AUC metric. It was trained for 100 epochs with a batch size of 4096.

## Model Performance

The model achieved the following ROC AUC scores:

*   Training Set: 0.705
*   Test Set: 0.699

## Requirements

To run the project, you will need the following Python packages:

*   pandas
*   numpy
*   scikit-learn
*   tensorflow
*   keras
*   plotly
