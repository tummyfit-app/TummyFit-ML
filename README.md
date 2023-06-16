## Tummyfit-ML

Project: TummyFit

Description:
The TummyFit Meal Planner is a Python-based application that generates a weekly meal plan based on user preferences and calorie requirements.
It uses a pre-trained TensorFlow model to suggest menu combinations and provides information about the nutritional content of the meals.

Usage:

1. Install the required dependencies by running the following command:
   pip install -r requirements.txt

2. Ensure that you have the necessary files:

- Trained model file ('tummyfit_model.json') in the 'model' directory.
- Model weights file ('tummyfit_model_weights.h5') in the 'model' directory.
- Food data file ('data-tummyfit-v6.csv') in the 'food_data' directory.

3. Update the input parameters:

- Open the 'inference.py' file and modify the following variables according to your preferences:
  - weight: Weight in kilograms (e.g., 70).
  - height: Height in centimeters (e.g., 170).
  - sex: Gender ('male' or 'female').
  - age: Age in years (e.g., 30).
  - daily_activity: Daily activity level ('sedentary', 'lightly active', 'moderately active', 'very active', 'extra active').
  - goal: Weight goal ('Maintain weight' or 'Weight loss').
  - halal: Set to True or False depending on your preference.
  - vegetarian: Set to True or False depending on your preference.
  - vegan: Set to True or False depending on your preference.
  - gluten_free: Set to True or False depending on your preference.
  - dairy_free: Set to True or False depending on your preference.

4. Run the application:

- Execute the following command to generate the weekly meal plan:
  ```
  python inference.py
  ```

5. View the output:

- The application will print the generated meal plan for each day of the week, along with the total calorie requirement.

Note: Make sure you have the necessary data files and dependencies before running the application.
