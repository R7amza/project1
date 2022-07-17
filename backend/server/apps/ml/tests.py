from cProfile import label
from django.test import TestCase

from apps.ml.startups_classifier.xgboost import XGBClassifier

class MLTests(TestCase):
    def test_xgb_algorithm(self):
        input_data = {
            "Estimated_Revenue_Range": 5500000, 
            "Total_Funding_Amount": 800000,
            "Headquarters_Location": "Riyadh", 
            "Founded_Date": 2018, 
            "Number_of_Founders": 1,
            "Funding_Stage": "Early Stage", 
            "Number_Funding_Rounds": 1, 
            "Number_of_Investors": 2,
            "Industry_Groups": "EduTech", 
            "Sector_Size": 4760000000, 
            "Number_of_Employees": 31,
            "Visit_Duration": 178, 
            "Bounce_Rate": 66, 
            "Monthly_Visits": 1606,
            "Monthly_Visits_Growth": -7
        }
        my_alg = XGBClassifier()
        response = my_alg.compute_prediction(input_data)
        self.assertEqual('ok', response['status'])
        self.assertTrue('label' in response)
        self.assertEqual(0 , response['label'])