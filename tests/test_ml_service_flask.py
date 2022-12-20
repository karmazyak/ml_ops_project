import os
from unittest import TestCase
from multifora_api import MlApi


class TestMlApi(TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["TEST"] = "1"

    def test_get(self):
        api = MlApi(host="http://<server ip>:<port>/")
        api.add_model("LG", "LogisticRegression", {})
        models_dict = api.gel_model_list().to_dict()
        with self.assertRaises(NotFound):
            gmr.get("wrong_type")

    def test_models_available(self):
        api = MlApi(host="http://<server ip>:<port>/")
        model_list = api.models_available()
        base_model_list = [
            "LogisticRegression",
            "RandomForestClassifier",
            "LinearRegression",
            "RandomForestRegressor",
            "DecisionTreeClassifier",
            "DecisionTreeRegressor",
            "KNeighborsClassifier",
            "KNeighborsRegressor",
        ]
        self.assertListEqual(model_list, base_model_list)


class TestGetHyperParamsRest(TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["TEST"] = "1"

    def test_get(self):
        ghpr = GetHyperParamsRest()

        self.assertIn("hyper_params", ghpr.get("LogisticRegression"))

        with self.assertRaises(NotFound):
            ghpr.get("LogisticRegressionWrong")
