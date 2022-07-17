from cProfile import label
from django.test import TestCase
# add at the beginning of the file:
import inspect
from apps.ml.registry import MLRegistry

from apps.ml.startups_classifier.xgboost1 import XGBClassifier1

class MLTests(TestCase):
    def test_xgb_algorithm(self):
        input_data = {
            'Estimated_Revenue_Range': 5500000, 
            'Total_Funding_Amount': 800000,
            'Headquarters_Location': "Riyadh", 
            'Founded_Date': 2018, 
            'Number_of_Founders': 1,
            'Funding_Stage': "Early Stage", 
            'Number_Funding_Rounds': 1, 
            'Number_of_Investors': 2,
            'Industry_Groups': "EduTech", 
            'Sector_Size': 4760000000, 
            'Number_of_Employees': 31,
            'Visit_Duration': 178, 
            'Bounce_Rate': 66, 
            'Monthly_Visits': 1606,
            'Monthly_Visits_Growth': -7
        }
        my_alg = XGBClassifier1()
        response = my_alg.compute_prediction(input_data)
        self.assertEqual('ok', response['status'])
        self.assertTrue('label' in response)
        self.assertEqual(1 , response['label'])

    def test_registry(self):
        registry = MLRegistry()
        self.assertEqual(len(registry.endpoints), 0)
        endpoint_name = "startups_classifier"
        algorithm_object = XGBClassifier1()
        algorithm_name = "XGBoost"
        algorithm_status = "production"
        algorithm_version = "0.0.1"
        algorithm_owner = "Reda"
        algorithm_description = "XGBoost with simple pre- and post-processing"
        algorithm_code = inspect.getsource(XGBClassifier1)
        # add to registry
        registry.add_algorithm(endpoint_name, algorithm_object, algorithm_name,
                    algorithm_status, algorithm_version, algorithm_owner,
                    algorithm_description, algorithm_code)
        # there should be one endpoint available
        self.assertEqual(len(registry.endpoints), 1)