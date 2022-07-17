# file backend/server/server/wsgi.py
import os
from django.core.wsgi import get_wsgi_application
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')
application = get_wsgi_application()

# ML registry
import inspect
from apps.ml.registry import MLRegistry
from apps.ml.startups_classifier.xgboost1 import XGBClassifier1

try:
    registry = MLRegistry() # create ML registry
    # Random Forest classifier
    xgb = XGBClassifier1()
    # add to ML registry
    registry.add_algorithm(endpoint_name="startups_classifier",
                            algorithm_object=xgb,
                            algorithm_name="XGBoost",
                            algorithm_status="production",
                            algorithm_version="0.0.1",
                            owner="Reda",
                            algorithm_description="Random Forest with simple pre- and post-processing",
                            algorithm_code=inspect.getsource(XGBClassifier1))

except Exception as e:
    print("Exception while loading the algorithms to the registry,", str(e))