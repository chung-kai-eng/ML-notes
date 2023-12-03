from typing import Optional, Union
import mlflow
from mlflow.entities import Experiment
from mlflow.entities.model_registry import ModelVersion
from mlflow.store.entities import PagedList
import request # for mlflow Authentication (account management)

# View more about pyfunc.PythonModel:
# https://mlflow.org/docs/latest/traditional-ml/creating-custom-pyfunc/notebooks/index.html?highlight=pyfunc
# ./tutorial_note/basic-pyfunc.ipynb
class MyModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        return self.my_custom_function(model_input, params)
    
    def my_custom_function(self, model_input, params=None):
        return 0


my_model = MyModel()

class MLFlowTool:
    def __init__(self, account: Optional[str] = None, password: Optional[str] = None):
        if account and password:
            self.login(account=account, password=password)
        else:
          mlflow.set_tracking_uri('')
        
        self.client = mlflow.MlflowClient()

    def login(self, account: str, password: str):
        response = requests.get(
            url='',
            auth=(account, password)
        )
        print(response)


    def add_model_record(self, UUID: str, stage: str, naming_1: str, naming_2: str, register_model: bool=False) -> None:
        model_name = self.name_exp_rule(naming_1, naming_2)
        mlflow.set_experiment(experiment_name=model_name)
        if mlflow.active_run():
            mlflow.end_run()
          
        with mlflow.start_run(run_name=UUID):
            mlflow.pyfunc.log_model(
                artifact_path='model',
                python_model=my_model,
                registered_model_name=model_name if register_model else None
            )
        if register_model:
            model_list: list[ModelVersion] = self.client.get_registered_model(name=model_name)
            # get the latest model version
            latest_version = model_list.latest_versions[-1]
            version = latest_version.version

            # set model version tag & stage (set both version and stage will result in an error)
            # set in key-value pair
            self.client.set_model_version_tag(name=model_name, version=version, key='', value='')
            self.client.transition_model_version_stage(name=model_name, version=version, stage=stage)
            # check the current stage and tags
            model_info = self.client.get_model_version(name=model_name, version=version)
            current_stage = model_info.current_stage
            tags = model_info.tags
            print(f'New Registed Model: ')
            print(f'Version: {version}')
            print(f'Model Stage: {stage}')
            print(f'Model tags: {tags}')
        return

    def query_model_with_tag(self, naming_1: str, naming_2: str) -> str:
        model_name = self.name_exp_rule(naming_1, naming_2)
        model_list: list[ModelVersion] = self.client.get_latest_versions(model_name, stages=['Production'])
        if len(model_list) != 1:
            raise ValueError('Only one production model is allowed')
        model_record: ModelVersion = model_list[0]
        model_tags = model_record.tags
        
        return model_tags['<key>']

    def delete_registered_model(self, model_name: str, version: Optional[int] = None) -> None:
        """Warning: if not specify version, all models in the model_name will be deleted"""
        if version is None:
            self.client.delete_registered_model(name=model_name)
        else:
            self.client.delete_model_version(name=model_name, version=version)

    def _search_condition(self, name: Optional[str] = None, tags: Optional[dict[str]] = None) -> str:
        filter_condition = ''
        if name is not None:
            filter_condition += f"name='name' and "
        
        if tags is not None:
            for key, value in tags.items():
                filter_condition += f"tags.{key}='{value}' and "
        
        filter_condition = filter_condition.removesuffix(' and ')
        print(filter_condition)
        return filter_condition
    
    def search_experiment(self, name: Optional[str], tags: Optional[dict[str]] = None,
                         return_all: bool = False) -> Union[PagedList[Experiment], Experiment]:
        filter_str = self._search_condition(name=name, tags=tags)
        exps = self.client.search_experiment(filter_str=filter_str)

        if len(exps) == 0:
            raise ModuleNotFoundError('There is no model under the condition')
        if return_all:
            return exps
        if len(exps) == 1:
            return exps[0]
        else:
            sorted_exps = sorted(exps, key=lambda x: x.creation_timestamp, reverse=True)
            return sorted_exps[0]

    def search_mode_version(self, name: Optional[str], tags: Optional[dict[str]] = None,
                            get_latest_model: bool = False, mdoel_stage: str = "Production",
                            return_all: bool = False) -> Union[PagedList[ModelVersion], ModelVersion]:
        filter_str = self._search_condition(name=name, tags=tags)
        models = self.client.search_model_versions(filter_str=filter_str)
        if len(models) == 0:
            raise ModuleNotFoundError('There is no model under the condition')
        
        if return_all:
            return models
        if len(exps) == 1:
            return models[0]
        else:
            sorted_model_versions = sorted(models, key=lambda x: x.creation_timestamp, reverse=True)
            model_version = next((m for m in sorted_model_versions if m.current_stage == model_stage), None)
            return model_version  

    @staticmethod
    def name_exp_rule(naming_1: str, naming_2: str) -> str:
        return f'{naming_1}_{naming_2}'
