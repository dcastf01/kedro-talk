from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import evaluate_model, split_data, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=['heart', 'params:model_options'],
                outputs=['X_train', 'X_test', 'y_train', 'y_test'],
                name='split_data_node',
            ),
            node(
                func=train_model,
                inputs=['X_train', 'y_train'],
                outputs='classifier',
                name='train_model_node',
            ),
            node(
                func=evaluate_model,
                inputs=['classifier', 'X_test', 'y_test'],
                name='evaluate_model_node',
                outputs='metrics',
            ),
        ],
        tags='02',
    )
