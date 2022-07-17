# file backend/server/endpoints/tests.py
from django.test import TestCase
from rest_framework.test import APIClient

class EndpointTests(TestCase):

    def test_predict_view(self):
        client = APIClient()
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
        classifier_url = "/api/v1/startups_classifier/predict"
        response = client.post(classifier_url, input_data, format='json')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["label"], 1)
        self.assertTrue("request_id" in response.data)
        self.assertTrue("status" in response.data)